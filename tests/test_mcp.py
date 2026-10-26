"""MCP connectors: catalog, store, proxy tools, registry sync, and the UI API.

Everything runs offline — no MCP server is ever spawned. The live-connection
paths (probe/call) are monkeypatched at the client layer.
"""

from __future__ import annotations

import pytest

from ai_employee.tools.mcp import (
    CONNECTOR_CATALOG,
    MCPProxyTool,
    RenderedServer,
    build_mcp_tools,
    connector_for,
    delete_server,
    load_servers,
    proxy_tool_name,
    render_server,
    save_server,
    sync_registry,
)

ECHO_TOOL = {
    "name": "echo",
    "description": "Echo a message back.",
    "input_schema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
}


# ── Catalog ──────────────────────────────────────────────────────────────────
def test_catalog_covers_popular_connectors():
    ids = {c.id for c in CONNECTOR_CATALOG}
    for expected in (
        "filesystem",
        "github",
        "slack",
        "postgres",
        "playwright",
        "git",
        "gmail",
        "google_calendar",
        "gdrive",
    ):
        assert expected in ids
    # Custom escape hatches for anything not in the catalog.
    assert "custom_stdio" in ids and "custom_http" in ids


def test_catalog_templates_render_cleanly():
    """Every {placeholder} in a connector's templates must be a declared param."""
    for spec in CONNECTOR_CATALOG:
        if spec.auth == "google":  # no server: backed by the native Google tools
            continue
        params = {p.name: p.default or "x" for p in spec.params}
        rendered = render_server(spec, params)
        assert rendered.transport in ("stdio", "http")
        if spec.transport == "stdio":
            assert rendered.command.strip()
        else:
            assert rendered.url.strip()


def test_render_requires_required_params():
    with pytest.raises(ValueError, match="access token"):
        render_server(connector_for("github"), {})


def test_render_drops_headers_with_blank_optional_params():
    spec = connector_for("custom_http")
    rendered = render_server(spec, {"url": "https://host/mcp", "bearer_token": ""})
    assert rendered.headers == {}
    with_token = render_server(spec, {"url": "https://host/mcp", "bearer_token": "tok"})
    assert with_token.headers == {"Authorization": "Bearer tok"}


# ── Store ────────────────────────────────────────────────────────────────────
def test_store_roundtrip(settings):
    assert load_servers(settings) == {}
    save_server(settings, "github", params={"token": "ghp_x"}, require_approval=True)
    save_server(settings, "github", tools=[ECHO_TOOL], connected_at="2026-08-26T00:00:00")
    entry = load_servers(settings)["github"]
    assert entry["enabled"] and entry["require_approval"]
    assert entry["params"] == {"token": "ghp_x"}  # partial updates keep other fields
    assert entry["tools"] == [ECHO_TOOL]
    assert delete_server(settings, "github")
    assert not delete_server(settings, "github")


# ── Proxy tools + registry ───────────────────────────────────────────────────
def test_build_mcp_tools_from_cache_and_registry_sync(settings, registry):
    save_server(settings, "fetch", params={}, tools=[ECHO_TOOL])
    tools = build_mcp_tools(settings)
    assert [t.name for t in tools] == ["mcp_fetch_echo"]
    spec = tools[0].to_spec()
    assert spec.parameters == ECHO_TOOL["input_schema"]  # server schema verbatim
    assert "[MCP:fetch]" in spec.description

    count = sync_registry(settings, registry)
    assert count == 1 and "mcp_fetch_echo" in registry
    # Disabled servers drop out on the next sync.
    save_server(settings, "fetch", enabled=False)
    assert sync_registry(settings, registry) == 0
    assert "mcp_fetch_echo" not in registry


def test_default_registry_loads_cached_mcp_tools(settings):
    from ai_employee.tools import default_registry

    save_server(settings, "memory", tools=[ECHO_TOOL])
    reg = default_registry(settings)
    assert "mcp_memory_echo" in reg


def test_proxy_tool_name_sanitized():
    assert proxy_tool_name("github", "create-issue!") == "mcp_github_create_issue_"


async def test_proxy_tool_executes_via_client(settings, monkeypatch):
    calls = {}

    async def fake_call(server, tool_name, arguments, *, timeout):
        calls.update(server=server.id, tool=tool_name, args=arguments)
        return {"ok": True, "output": "pong"}

    monkeypatch.setattr("ai_employee.tools.mcp.proxy.call_server_tool", fake_call)
    tool = MCPProxyTool(
        RenderedServer(id="fetch", transport="stdio", command="uvx mcp-server-fetch"),
        "echo",
        "Echo",
        ECHO_TOOL["input_schema"],
    )
    result = await tool.execute(message="ping", extra_arg=1)
    assert result.ok and result.output == "pong"
    assert calls == {"server": "fetch", "tool": "echo", "args": {"message": "ping", "extra_arg": 1}}


async def test_proxy_tool_approval_gate(settings, registry):
    from ai_employee.core.exceptions import ApprovalRequired

    save_server(settings, "fetch", tools=[ECHO_TOOL], require_approval=True)
    sync_registry(settings, registry)
    with pytest.raises(ApprovalRequired):
        await registry.execute("mcp_fetch_echo", {"message": "hi"})


# ── UI API ───────────────────────────────────────────────────────────────────
async def test_mcp_options_endpoint_masks_secrets(settings, client):
    save_server(settings, "github", params={"token": "ghp_secret1234"}, tools=[ECHO_TOOL])
    resp = await client.get("/ui/api/mcp/options")
    assert resp.status_code == 200
    data = resp.json()
    assert {"sdk", "npx", "uvx"} <= set(data["runtime"])
    github = next(c for c in data["connectors"] if c["id"] == "github")
    assert github["configured"] and github["tools"] == ["echo"]
    token = next(p for p in github["params"] if p["name"] == "token")
    assert token["configured"] and "ghp_secret1234" not in str(token)
    assert token["masked"].startswith("••••")


async def test_mcp_add_refresh_toggle_delete_endpoints(settings, client, monkeypatch):
    async def fake_probe(server, *, timeout):
        return [ECHO_TOOL]

    monkeypatch.setattr("ai_employee.tools.mcp.manager.probe_server", fake_probe)

    resp = await client.put("/ui/api/mcp/servers/fetch", json={"params": {}})
    body = resp.json()
    assert resp.status_code == 200 and body["ok"] and body["registered_tools"] == 1

    resp = await client.post("/ui/api/mcp/servers/fetch/refresh")
    assert resp.status_code == 200 and resp.json()["ok"]

    resp = await client.post("/ui/api/mcp/servers/fetch/toggle", json={"enabled": False})
    assert resp.status_code == 200 and resp.json()["registered_tools"] == 0

    resp = await client.delete("/ui/api/mcp/servers/fetch")
    assert resp.status_code == 200
    assert load_servers(settings) == {}

    assert (await client.put("/ui/api/mcp/servers/nope", json={"params": {}})).status_code == 404
    assert (await client.delete("/ui/api/mcp/servers/fetch")).status_code == 404


async def test_mcp_connect_failure_recorded(settings, client, monkeypatch):
    async def fake_probe(server, *, timeout):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr("ai_employee.tools.mcp.manager.probe_server", fake_probe)
    resp = await client.put("/ui/api/mcp/servers/fetch", json={"params": {}})
    body = resp.json()
    assert resp.status_code == 200 and not body["ok"]
    assert "spawn failed" in body["error"]
    assert "spawn failed" in load_servers(settings)["fetch"]["error"]


# ── OAuth (browser sign-in) connectors ───────────────────────────────────────
def _mock_oauth_client(requests_seen: list):
    """An httpx client whose transport plays a full OAuth 2.1 provider."""
    import json as _json

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        path = request.url.path
        if path.startswith("/.well-known/oauth-protected-resource"):
            return httpx.Response(200, json={"authorization_servers": ["https://auth.example.com"]})
        if path.startswith("/.well-known/oauth-authorization-server"):
            return httpx.Response(
                200,
                json={
                    "authorization_endpoint": "https://auth.example.com/oauth/authorize",
                    "token_endpoint": "https://auth.example.com/oauth/token",
                    "registration_endpoint": "https://auth.example.com/oauth/register",
                    "scopes_supported": ["read", "write"],
                },
            )
        if path == "/oauth/register":
            body = _json.loads(request.content)
            assert "http" in body["redirect_uris"][0]
            return httpx.Response(201, json={"client_id": "dyn-client-1"})
        if path == "/oauth/token":
            form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
            if form["grant_type"] == "authorization_code":
                assert form["code"] == "the-code" and "code_verifier" in form
                return httpx.Response(
                    200,
                    json={
                        "access_token": "at-1",
                        "refresh_token": "rt-1",
                        "expires_in": 3600,
                    },
                )
            assert form["grant_type"] == "refresh_token" and form["refresh_token"] == "rt-1"
            return httpx.Response(200, json={"access_token": "at-2", "expires_in": 3600})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_catalog_oauth_connectors_need_no_params():
    for cid in ("notion", "linear", "sentry", "atlassian", "asana", "zapier"):
        spec = connector_for(cid)
        assert spec.auth == "oauth" and spec.transport == "http"
        assert not any(p.required for p in spec.params)
        assert render_server(spec, {}).url.startswith("https://")


async def test_oauth_discovery_falls_back_to_defaults():
    import httpx

    from ai_employee.tools.mcp.oauth import discover_metadata

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    meta = await discover_metadata("https://mcp.example.com/mcp", client=client)
    assert meta["authorization_endpoint"] == "https://mcp.example.com/authorize"
    assert meta["token_endpoint"] == "https://mcp.example.com/token"


async def test_oauth_begin_complete_and_refresh(settings):
    from urllib.parse import parse_qs, urlparse

    from ai_employee.tools.mcp.oauth import (
        begin_authorization,
        complete_authorization,
        ensure_access_token,
    )

    seen: list = []
    client = _mock_oauth_client(seen)

    url = await begin_authorization(
        settings,
        "linear",
        "https://mcp.example.com/mcp",
        "http://localhost:8000/ui/api/mcp/oauth/callback",
        client=client,
    )
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/oauth/authorize"
    assert q["client_id"] == ["dyn-client-1"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["resource"] == ["https://mcp.example.com/mcp"]

    server_id = await complete_authorization(settings, q["state"][0], "the-code", client=client)
    assert server_id == "linear"
    auth = load_servers(settings)["linear"]["auth"]
    assert auth["access_token"] == "at-1" and auth["refresh_token"] == "rt-1"

    # Valid token → returned as-is (no network hit).
    assert await ensure_access_token(settings, "linear", client=client) == "at-1"

    # Expired token → transparently refreshed and persisted.
    save_server(settings, "linear", auth={**auth, "expires_at": 0})
    assert await ensure_access_token(settings, "linear", client=client) == "at-2"
    assert load_servers(settings)["linear"]["auth"]["access_token"] == "at-2"
    # Refresh token was not rotated by the provider — the old one is kept.
    assert load_servers(settings)["linear"]["auth"]["refresh_token"] == "rt-1"


async def test_oauth_connect_requires_authorization_first(settings, monkeypatch):
    from ai_employee.tools.mcp import connect_server

    async def boom(server, *, timeout):
        raise AssertionError("must not probe before the user authorizes")

    monkeypatch.setattr("ai_employee.tools.mcp.manager.probe_server", boom)
    result = await connect_server(settings, "notion")
    assert result["needs_auth"] and not result["ok"]


async def test_oauth_proxy_injects_fresh_token(settings, monkeypatch):
    save_server(
        settings,
        "notion",
        tools=[ECHO_TOOL],
        auth={"access_token": "tok-live", "expires_at": 9e12, "token_endpoint": "x"},
    )
    tools = build_mcp_tools(settings)
    captured = {}

    async def fake_call(server, tool_name, arguments, *, timeout):
        captured["headers"] = server.headers
        return {"ok": True, "output": "done"}

    monkeypatch.setattr("ai_employee.tools.mcp.proxy.call_server_tool", fake_call)
    result = await tools[0].execute(message="hi")
    assert result.ok
    assert captured["headers"]["Authorization"] == "Bearer tok-live"
