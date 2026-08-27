"""OAuth 2.1 client for remote MCP servers — "click Connect, approve, done".

Implements the MCP authorization spec the same way Claude / Cursor connect to
remote servers like Notion, Linear, or Zapier:

1. **Discovery** — find the server's authorization server via RFC 9728
   (protected-resource metadata) and its endpoints via RFC 8414 / OIDC
   discovery, with spec-default fallbacks.
2. **Dynamic client registration** (RFC 7591) — no pre-created API keys; the
   framework registers itself as a public client on the fly.
3. **Authorization-code + PKCE** — the user approves in their browser; the
   provider redirects back to the console's callback route.
4. **Token exchange & refresh** — tokens live in the MCP store (chmod 600) and
   are refreshed transparently before tool calls.

Everything is plain ``httpx`` — no network happens at import or startup, so the
offline test suite stays green.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Any
from urllib.parse import urlencode, urlparse

from ai_employee.core.exceptions import ToolError
from ai_employee.core.logging import get_logger
from ai_employee.tools.mcp.store import load_servers, save_server

log = get_logger("mcp.oauth")

CLIENT_NAME = "AI Employee Framework"
_HTTP_TIMEOUT = 20.0

# In-flight authorization flows keyed by the OAuth ``state`` value. They only
# need to survive between "open browser" and "callback hits us" in one process.
_PENDING: dict[str, dict[str, Any]] = {}
_MAX_PENDING = 20


def _client(**kwargs: Any):
    import httpx

    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True, **kwargs)


def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


async def discover_metadata(server_url: str, *, client: Any = None) -> dict[str, Any]:
    """Resolve the authorization-server metadata for a remote MCP server.

    Tries RFC 9728 protected-resource metadata to find the authorization
    server, then RFC 8414 / OIDC discovery for its endpoints. Falls back to
    the MCP-spec default paths (``/authorize``, ``/token``, ``/register``)
    rooted at the server origin when discovery documents are missing.
    """
    own_client = client is None
    client = client or _client()
    try:
        origin = _origin(server_url)
        path = urlparse(server_url).path.rstrip("/")

        auth_server = origin
        scopes: list[str] = []
        for probe in (
            f"{origin}/.well-known/oauth-protected-resource{path}",
            f"{origin}/.well-known/oauth-protected-resource",
        ):
            data = await _get_json(client, probe)
            if data and data.get("authorization_servers"):
                auth_server = str(data["authorization_servers"][0]).rstrip("/")
                scopes = list(data.get("scopes_supported") or [])
                break

        as_origin = _origin(auth_server)
        as_path = urlparse(auth_server).path.rstrip("/")
        for probe in (
            f"{as_origin}/.well-known/oauth-authorization-server{as_path}",
            f"{as_origin}/.well-known/oauth-authorization-server",
            f"{as_origin}/.well-known/openid-configuration{as_path}",
            f"{as_origin}/.well-known/openid-configuration",
        ):
            data = await _get_json(client, probe)
            if data and data.get("authorization_endpoint"):
                if not scopes:
                    scopes = list(data.get("scopes_supported") or [])
                return {**data, "_scopes": scopes}

        # Spec-default endpoints when no discovery document exists.
        return {
            "authorization_endpoint": f"{auth_server}/authorize",
            "token_endpoint": f"{auth_server}/token",
            "registration_endpoint": f"{auth_server}/register",
            "_scopes": scopes,
        }
    finally:
        if own_client:
            await client.aclose()


async def _get_json(client: Any, url: str) -> dict[str, Any] | None:
    try:
        resp = await client.get(url, headers={"Accept": "application/json"})
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - discovery probes are best-effort
        log.debug("mcp.oauth.probe_failed", url=url, error=str(exc))
    return None


async def _register_client(
    client: Any, metadata: dict[str, Any], redirect_uri: str
) -> dict[str, str]:
    """Dynamically register as a public client (RFC 7591)."""
    endpoint = metadata.get("registration_endpoint")
    if not endpoint:
        raise ToolError(
            "this server's authorization provider does not support dynamic client "
            "registration — use its API-key/custom connector instead"
        )
    resp = await client.post(
        endpoint,
        json={
            "client_name": CLIENT_NAME,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
        headers={"Accept": "application/json"},
    )
    if resp.status_code not in (200, 201):
        raise ToolError(f"client registration failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    return {
        "client_id": data["client_id"],
        "client_secret": data.get("client_secret", ""),
    }


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


async def begin_authorization(
    settings: Any,
    server_id: str,
    server_url: str,
    redirect_uri: str,
    *,
    client: Any = None,
) -> str:
    """Start the browser flow; returns the URL to open for user consent."""
    own_client = client is None
    client = client or _client()
    try:
        metadata = await discover_metadata(server_url, client=client)

        # Reuse a prior dynamic registration when it targeted the same
        # redirect URI; otherwise register a fresh public client.
        stored_auth = (load_servers(settings).get(server_id) or {}).get("auth") or {}
        if stored_auth.get("client_id") and stored_auth.get("redirect_uri") == redirect_uri:
            registration = {
                "client_id": stored_auth["client_id"],
                "client_secret": stored_auth.get("client_secret", ""),
            }
        else:
            registration = await _register_client(client, metadata, redirect_uri)

        verifier, challenge = _pkce_pair()
        state = secrets.token_urlsafe(32)
        while len(_PENDING) >= _MAX_PENDING:  # bound memory; oldest flow dies
            _PENDING.pop(next(iter(_PENDING)))
        _PENDING[state] = {
            "server_id": server_id,
            "server_url": server_url,
            "redirect_uri": redirect_uri,
            "verifier": verifier,
            "token_endpoint": metadata["token_endpoint"],
            "started_at": time.time(),
            **registration,
        }

        query = {
            "response_type": "code",
            "client_id": registration["client_id"],
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": server_url,
        }
        scopes = metadata.get("_scopes") or []
        if scopes:
            query["scope"] = " ".join(scopes)
        url = f"{metadata['authorization_endpoint']}?{urlencode(query)}"
        log.info("mcp.oauth.begin", server=server_id)
        return url
    finally:
        if own_client:
            await client.aclose()


async def complete_authorization(
    settings: Any, state: str, code: str, *, client: Any = None
) -> str:
    """Exchange the callback's code for tokens and persist them. Returns server id."""
    flow = _PENDING.pop(state, None)
    if flow is None:
        raise ToolError("unknown or expired authorization flow — click Connect again")

    own_client = client is None
    client = client or _client()
    try:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": flow["redirect_uri"],
            "client_id": flow["client_id"],
            "code_verifier": flow["verifier"],
            "resource": flow["server_url"],
        }
        if flow.get("client_secret"):
            form["client_secret"] = flow["client_secret"]
        resp = await client.post(
            flow["token_endpoint"], data=form, headers={"Accept": "application/json"}
        )
        if resp.status_code != 200:
            raise ToolError(f"token exchange failed ({resp.status_code}): {resp.text[:300]}")
        tokens = resp.json()
    finally:
        if own_client:
            await client.aclose()

    save_server(
        settings,
        flow["server_id"],
        auth={
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token", ""),
            "expires_at": _expires_at(tokens),
            "token_endpoint": flow["token_endpoint"],
            "client_id": flow["client_id"],
            "client_secret": flow.get("client_secret", ""),
            "redirect_uri": flow["redirect_uri"],
            "resource": flow["server_url"],
        },
    )
    log.info("mcp.oauth.completed", server=flow["server_id"])
    return str(flow["server_id"])


def _expires_at(tokens: dict[str, Any]) -> float:
    try:
        return time.time() + float(tokens.get("expires_in") or 3600)
    except (TypeError, ValueError):
        return time.time() + 3600


async def ensure_access_token(
    settings: Any, server_id: str, *, client: Any = None
) -> str | None:
    """A valid access token for the server, refreshing if it is (nearly) expired.

    Returns ``None`` when the server was never authorized. Raises
    :class:`ToolError` when the stored grant is dead and the user must
    re-approve in the browser.
    """
    auth = (load_servers(settings).get(server_id) or {}).get("auth") or {}
    token = auth.get("access_token")
    if not token:
        return None
    if time.time() < float(auth.get("expires_at") or 0) - 60:
        return str(token)
    if not auth.get("refresh_token"):
        raise ToolError(f"authorization for '{server_id}' expired — reconnect it from 🔌 MCP")

    own_client = client is None
    client = client or _client()
    try:
        form = {
            "grant_type": "refresh_token",
            "refresh_token": auth["refresh_token"],
            "client_id": auth.get("client_id", ""),
        }
        if auth.get("client_secret"):
            form["client_secret"] = auth["client_secret"]
        if auth.get("resource"):
            form["resource"] = auth["resource"]
        resp = await client.post(
            auth["token_endpoint"], data=form, headers={"Accept": "application/json"}
        )
        if resp.status_code != 200:
            raise ToolError(
                f"token refresh for '{server_id}' failed — reconnect it from 🔌 MCP "
                f"({resp.status_code})"
            )
        tokens = resp.json()
    finally:
        if own_client:
            await client.aclose()

    new_auth = {
        **auth,
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token") or auth["refresh_token"],
        "expires_at": _expires_at(tokens),
    }
    save_server(settings, server_id, auth=new_auth)
    log.info("mcp.oauth.refreshed", server=server_id)
    return str(tokens["access_token"])


__all__ = [
    "begin_authorization",
    "complete_authorization",
    "discover_metadata",
    "ensure_access_token",
]
