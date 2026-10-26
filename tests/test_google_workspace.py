"""Google OAuth (per-user Gmail/Calendar consent) + the native Google tools.

Fully offline: Google's endpoints are played by an httpx MockTransport injected
through each function's ``client=`` parameter, and tool API calls are stubbed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from ai_employee.integrations.google_oauth import (
    begin_authorization,
    client_config,
    complete_authorization,
    ensure_access_token,
    load_accounts,
    remove_account,
    resolve_account,
)
from ai_employee.tools.builtins.google_workspace import build_google_tools

REDIRECT = "http://localhost:8000/ui/api/google/oauth/callback"


def _write_client_file(settings, kind: str = "installed") -> None:
    Path(settings.google_oauth_client_file).write_text(
        json.dumps(
            {
                kind: {
                    "client_id": "op-client-id.apps.googleusercontent.com",
                    "client_secret": "op-secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
        )
    )


def _fake_id_token(email: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"email": email}).encode()).rstrip(b"=")
    return f"h.{payload.decode()}.sig"


def _mock_google(seen: list):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "oauth2.googleapis.com" and request.url.path == "/token":
            form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
            if form["grant_type"] == "authorization_code":
                assert form["code"] == "g-code" and "code_verifier" in form
                return httpx.Response(
                    200,
                    json={
                        "access_token": "ga-1",
                        "refresh_token": "gr-1",
                        "expires_in": 3600,
                        "id_token": _fake_id_token("user@example.com"),
                        "scope": "openid email",
                    },
                )
            assert form["grant_type"] == "refresh_token"
            return httpx.Response(200, json={"access_token": "ga-2", "expires_in": 3600})
        if request.url.path == "/revoke":
            return httpx.Response(200)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── Client config ────────────────────────────────────────────────────────────
def test_client_config_absent_and_present(settings):
    assert client_config(settings) is None
    _write_client_file(settings, "web")
    cfg = client_config(settings)
    assert cfg["client_id"].startswith("op-client-id")
    assert cfg["token_uri"] == "https://oauth2.googleapis.com/token"


def test_tools_hidden_until_configured(settings):
    assert build_google_tools(settings) == []
    _write_client_file(settings)
    names = {t.name for t in build_google_tools(settings)}
    assert names == {
        "gmail_search",
        "gmail_read",
        "gmail_send",
        "calendar_list_events",
        "calendar_create_event",
        "drive_search",
        "drive_read",
    }


# ── Consent flow ─────────────────────────────────────────────────────────────
async def test_full_consent_flow_and_refresh(settings):
    _write_client_file(settings)
    url = begin_authorization(settings, REDIRECT)
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).netloc == "accounts.google.com"
    assert q["access_type"] == ["offline"] and q["prompt"] == ["consent"]
    assert q["code_challenge_method"] == ["S256"]
    assert "gmail.readonly" in q["scope"][0]

    seen: list = []
    client = _mock_google(seen)
    email = await complete_authorization(settings, q["state"][0], "g-code", client=client)
    assert email == "user@example.com"
    entry = load_accounts(settings)[email]
    assert entry["access_token"] == "ga-1" and entry["refresh_token"] == "gr-1"

    # Fresh token → returned without a network round-trip.
    assert await ensure_access_token(settings, email, client=client) == "ga-1"

    # Expired → refreshed and persisted; un-rotated refresh token is kept.
    from ai_employee.integrations.google_oauth import _save_accounts

    entry["expires_at"] = 0
    accounts = load_accounts(settings)
    accounts[email] = entry
    _save_accounts(settings, accounts)
    assert await ensure_access_token(settings, email, client=client) == "ga-2"
    assert load_accounts(settings)[email]["refresh_token"] == "gr-1"

    # Disconnect revokes at Google and forgets the account.
    assert await remove_account(settings, email, client=_mock_google([]))
    assert load_accounts(settings) == {}


async def test_unknown_state_rejected(settings):
    _write_client_file(settings)
    from ai_employee.core.exceptions import ToolError

    with pytest.raises(ToolError, match="expired"):
        await complete_authorization(settings, "bogus-state", "code", client=_mock_google([]))


def test_resolve_account_rules(settings):
    from ai_employee.core.exceptions import ToolError

    with pytest.raises(ToolError, match="no Google account"):
        resolve_account(settings)
    Path(settings.google_accounts_file).write_text(
        json.dumps({"accounts": {"a@x.com": {}, "b@x.com": {}}})
    )
    assert resolve_account(settings, "a@x.com") == "a@x.com"
    with pytest.raises(ToolError, match="several"):
        resolve_account(settings)
    with pytest.raises(ToolError, match="not connected"):
        resolve_account(settings, "c@x.com")


# ── Tools ────────────────────────────────────────────────────────────────────
async def test_gmail_search_formats_messages(settings, monkeypatch):
    _write_client_file(settings)
    tool = next(t for t in build_google_tools(settings) if t.name == "gmail_search")

    async def fake_api(_s, _acct, method, url, *, params=None, json_body=None):
        if url.endswith("/messages"):
            assert params["q"] == "is:unread"
            return {"messages": [{"id": "m1"}]}
        assert "/messages/m1" in url
        return {
            "snippet": "hello there",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@x.com"},
                    {"name": "Subject", "value": "Hi"},
                    {"name": "Date", "value": "Wed, 27 Aug 2026"},
                ]
            },
        }

    monkeypatch.setattr("ai_employee.tools.builtins.google_workspace._api", fake_api)
    result = await tool.execute(query="is:unread")
    assert result.ok and "alice@x.com" in result.output and "[m1]" in result.output


async def test_gmail_send_is_approval_gated(settings, registry):
    from ai_employee.core.exceptions import ApprovalRequired

    _write_client_file(settings)
    for t in build_google_tools(settings):
        registry.register(t, replace=True)
    with pytest.raises(ApprovalRequired):
        await registry.execute("gmail_send", {"to": "b@x.com", "subject": "s", "body": "hi"})
    with pytest.raises(ApprovalRequired):
        await registry.execute(
            "calendar_create_event",
            {"summary": "standup", "start": "2026-08-28T09:00:00Z", "end": "2026-08-28T09:15:00Z"},
        )


# ── UI API ───────────────────────────────────────────────────────────────────
async def test_google_status_and_authorize_endpoints(settings, client):
    resp = await client.get("/ui/api/google/status")
    body = resp.json()
    assert resp.status_code == 200 and not body["configured"] and body["accounts"] == []

    # Not configured → authorize fails with a helpful message.
    resp = await client.post("/ui/api/google/authorize")
    assert resp.status_code == 502 and "client_secret.json" in resp.json()["detail"]

    _write_client_file(settings)
    resp = await client.post("/ui/api/google/authorize")
    assert resp.status_code == 200
    assert resp.json()["auth_url"].startswith("https://accounts.google.com/")
    assert (await client.get("/ui/api/google/status")).json()["configured"]

    resp = await client.delete("/ui/api/google/accounts/nobody@x.com")
    assert resp.status_code == 404
