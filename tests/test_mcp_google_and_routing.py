"""Google-backed MCP cards, MCP tool routing to agents, readable MCP errors,
Drive tools, and "Connect Google also signs a guest in".

Offline: Google is a stubbed account store / MockTransport; the only process
spawned is a tiny local Python script standing in for a crashing MCP server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_employee.tools.mcp import (
    MCPProxyTool,
    RenderedServer,
    build_mcp_tools,
    connector_for,
    load_servers,
    save_server,
    sync_registry,
)
from ai_employee.tools.mcp.client import describe_error, probe_server, sdk_available
from ai_employee.tools.mcp.manager import (
    connect_server,
    connected_services_by_agent,
    google_connector_state,
)

GMAIL = (
    "https://www.googleapis.com/auth/gmail.readonly " "https://www.googleapis.com/auth/gmail.send"
)
CALENDAR = "https://www.googleapis.com/auth/calendar"
DRIVE = "https://www.googleapis.com/auth/drive.readonly"
ECHO_TOOL = {"name": "echo", "description": "Echo.", "input_schema": {"type": "object"}}


def _client_file(settings) -> None:
    Path(settings.google_oauth_client_file).write_text(
        json.dumps({"installed": {"client_id": "cid", "client_secret": "sec"}})
    )


def _account(settings, email: str = "me@gmail.com", scopes: str = f"{GMAIL} {CALENDAR}") -> None:
    Path(settings.google_accounts_file).write_text(
        json.dumps(
            {
                "accounts": {
                    email: {
                        "access_token": "at",
                        "refresh_token": "rt",
                        "expires_at": 9e12,
                        "scopes": scopes,
                        "name": "Me",
                    }
                }
            }
        )
    )


# ── Google-backed cards ──────────────────────────────────────────────────────
def test_google_cards_need_no_server_or_params():
    for cid in ("gmail", "google_calendar", "gdrive"):
        spec = connector_for(cid)
        assert spec.auth == "google" and spec.transport == "google"
        assert not spec.params and not spec.command and not spec.url
        assert spec.google_scopes and spec.native_tools


def test_google_card_state_follows_the_connected_account(settings):
    gmail, drive = connector_for("gmail"), connector_for("gdrive")
    assert google_connector_state(settings, gmail)["configured"] is False

    _client_file(settings)
    state = google_connector_state(settings, gmail)
    assert state["configured"] and not state["ready"] and state["connected_accounts"] == []

    _account(settings)  # granted Gmail + Calendar, not Drive
    state = google_connector_state(settings, gmail)
    assert state["ready"] and state["accounts"] == ["me@gmail.com"]
    assert state["tools"] == ["gmail_search", "gmail_read", "gmail_send"]
    drive_state = google_connector_state(settings, drive)
    assert not drive_state["ready"] and drive_state["missing_scopes"] == [DRIVE]


async def test_connect_google_card_uses_account_and_drops_stale_server_entry(settings, monkeypatch):
    _client_file(settings)
    # A pre-existing third-party Gmail MCP entry (the old, key-file-based server).
    save_server(settings, "gmail", error="ExceptionGroup: unhandled errors in a TaskGroup")

    result = await connect_server(settings, "gmail")
    assert result == {
        "ok": False,
        "server": "gmail",
        "tools": [],
        "needs_google": True,
        "missing_scopes": [],
    }
    assert "gmail" not in load_servers(settings)

    _account(settings)
    probed: list[tuple[str, str]] = []

    async def fake_probe(s, server_id, account):
        probed.append((server_id, account))
        return None

    monkeypatch.setattr("ai_employee.tools.mcp.manager.probe_google_service", fake_probe)
    result = await connect_server(settings, "gmail")
    assert result["ok"] and result["tools"] == ["gmail_search", "gmail_read", "gmail_send"]
    assert probed == [("gmail", "me@gmail.com")]
    assert load_servers(settings) == {}  # nothing to store or spawn

    result = await connect_server(settings, "gdrive")
    assert not result["ok"] and result["needs_google"] and result["missing_scopes"] == [DRIVE]


async def test_connect_google_card_without_client_file_explains_setup(settings):
    result = await connect_server(settings, "google_calendar")
    assert not result["ok"] and "OAuth client" in result["error"]


def test_stale_google_entries_never_become_proxy_tools(settings, registry):
    save_server(settings, "gmail", tools=[ECHO_TOOL])
    assert build_mcp_tools(settings) == []
    assert sync_registry(settings, registry) == 0


async def test_mcp_options_reports_google_cards(settings, client):
    _client_file(settings)
    _account(settings)
    data = (await client.get("/ui/api/mcp/options")).json()
    cards = {c["id"]: c for c in data["connectors"]}
    gmail = cards["gmail"]
    assert gmail["auth"] == "google" and gmail["configured"] and gmail["enabled"]
    assert gmail["google"]["accounts"] == ["me@gmail.com"]
    assert gmail["agents"] == ["comms"]
    assert cards["gdrive"]["configured"] is False
    assert cards["gdrive"]["google"]["missing_scopes"] == [DRIVE]
    assert cards["slack"]["agents"] == ["comms"]
    assert cards["time"]["agents"] == []  # every worker


async def test_put_google_card_reports_needs_google(settings, client):
    _client_file(settings)
    res = (await client.put("/ui/api/mcp/servers/gdrive", json={"params": {}})).json()
    assert res["ok"] is False and res["needs_google"] is True


# ── MCP tools reach the agents they are routed to ────────────────────────────
def _proxy(server_id: str, agents: tuple[str, ...]) -> MCPProxyTool:
    return MCPProxyTool(
        RenderedServer(id=server_id, transport="stdio", command="x"),
        "echo",
        "Echo",
        None,
        agents=agents,
    )


def test_agents_see_only_mcp_tools_routed_to_them(registry):
    from ai_employee.agents import build_agent_registry

    registry.register(_proxy("slack", ("comms",)))
    registry.register(_proxy("time", ()))  # empty routing = every worker
    agents = build_agent_registry()
    ctx = SimpleNamespace(tools=registry, denied_tools=frozenset())

    def exposed(name: str) -> set[str]:
        return {spec.name for spec in agents.get(name)._exposed_specs(ctx)}

    assert {"mcp_slack_echo", "mcp_time_echo"} <= exposed("comms")
    assert "mcp_slack_echo" not in exposed("coding")
    assert "mcp_time_echo" in exposed("coding")
    assert "mcp_time_echo" in exposed("research")

    ctx.denied_tools = frozenset({"mcp_time_echo"})
    assert "mcp_time_echo" not in exposed("research")


def test_built_proxy_tools_carry_catalog_routing(settings):
    save_server(settings, "slack", params={"bot_token": "x", "team_id": "T"}, tools=[ECHO_TOOL])
    (tool,) = build_mcp_tools(settings)
    assert tool.agents == ("comms",) and tool.routes_to("comms") and not tool.routes_to("coding")


def test_comms_available_through_a_comms_mcp_server(settings):
    from ai_employee.agents import build_agent_registry
    from ai_employee.agents.registry import INTEGRATION_CHECKS

    assert INTEGRATION_CHECKS["comms"]() is False  # no Google, no MCP
    assert "comms" not in {a["name"] for a in build_agent_registry().catalog()}

    save_server(settings, "slack", params={"bot_token": "x", "team_id": "T"}, tools=[ECHO_TOOL])
    assert connected_services_by_agent(settings) == {"comms": ["Slack"]}
    assert INTEGRATION_CHECKS["comms"]() is True
    roster = {a["name"]: a for a in build_agent_registry().catalog()}
    assert "Connected services: Slack." in roster["comms"]["description"]
    assert "Connected services" not in roster["coding"]["description"]


def test_comms_available_through_google(settings):
    from ai_employee.agents.registry import INTEGRATION_CHECKS

    _client_file(settings)
    _account(settings)
    assert INTEGRATION_CHECKS["comms"]() is True


# ── Readable MCP errors ──────────────────────────────────────────────────────
def test_describe_error_unwraps_groups_and_adds_server_output():
    exc = ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("Connection closed")])
    message = describe_error(exc, "Error: OAuth keys file not found")
    assert "ExceptionGroup" not in message and "TaskGroup" not in message
    assert message == (
        "RuntimeError: Connection closed — server said: Error: OAuth keys file not found"
    )
    assert "npx/uvx" in describe_error(FileNotFoundError(2, "No such file or directory"))


@pytest.mark.skipif(not sdk_available(), reason="mcp package not installed")
async def test_crashing_stdio_server_reports_its_own_message():
    script = "import sys; sys.stderr.write('Error: OAuth keys file not found\\n'); sys.exit(1)"
    server = RenderedServer(
        id="broken",
        transport="stdio",
        command=f'{sys.executable} -c "{script}"',
    )
    from ai_employee.core.exceptions import ToolError

    with pytest.raises(ToolError) as info:
        await probe_server(server, timeout=20)
    assert "OAuth keys file not found" in str(info.value)
    assert "TaskGroup" not in str(info.value)


async def test_failed_connect_records_readable_error(settings, monkeypatch):
    async def boom(server, *, timeout):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [OSError("spawn failed")])

    monkeypatch.setattr("ai_employee.tools.mcp.manager.probe_server", boom)
    result = await connect_server(settings, "fetch")
    assert result["error"] == "OSError: spawn failed"
    assert load_servers(settings)["fetch"]["error"] == "OSError: spawn failed"


# ── Drive tools ──────────────────────────────────────────────────────────────
def test_drive_query_plain_words_vs_syntax():
    from ai_employee.tools.builtins.google_workspace import _drive_query

    assert _drive_query("") == "trashed = false"
    assert _drive_query("q3 budget") == "fullText contains 'q3 budget' and trashed = false"
    assert _drive_query("it's") == "fullText contains 'it\\'s' and trashed = false"
    syntax = "name contains 'plan'"
    assert _drive_query(syntax) == f"({syntax}) and trashed = false"


async def test_drive_read_exports_docs_and_describes_binaries(settings, monkeypatch):
    from ai_employee.tools.builtins import google_workspace as gw

    calls: list[tuple[str, dict]] = []
    files = {
        "doc1": {"name": "Plan", "mimeType": "application/vnd.google-apps.document"},
        "pdf1": {"name": "Scan", "mimeType": "application/pdf"},
    }

    async def fake_api(settings, account, method, url, *, params=None, json_body=None, raw=False):
        calls.append((url, params or {}))
        if raw:
            return SimpleNamespace(text="Hello from the doc")
        return files[url.rsplit("/", 1)[-1]]

    monkeypatch.setattr(gw, "_api", fake_api)
    tool = gw.DriveReadTool(settings)
    result = await tool.execute(file_id="doc1")
    assert result.ok and "Hello from the doc" in result.output
    assert calls[-1][0].endswith("/files/doc1/export") and calls[-1][1] == {
        "mimeType": "text/plain"
    }

    result = await tool.execute(file_id="pdf1")
    assert not result.ok and "binary file" in result.output


async def test_google_api_insufficient_scope_says_how_to_fix(settings, monkeypatch):
    import httpx

    from ai_employee.core.exceptions import ToolError
    from ai_employee.tools.builtins import google_workspace as gw

    _client_file(settings)
    _account(settings)
    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        body = {
            "error": {"code": 403, "message": "Request had insufficient authentication scopes."}
        }
        transport = httpx.MockTransport(lambda r: httpx.Response(403, json=body))
        return real_client(transport=transport)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    with pytest.raises(ToolError, match="Google Drive card"):
        await gw._api(settings, "", "GET", f"{gw._DRIVE}/files")


# ── Connect Google also signs a guest in ─────────────────────────────────────
async def test_google_connect_signs_a_guest_in(settings, client, monkeypatch):
    _client_file(settings)

    async def fake_complete(s, state, code):
        _account(s, "new@gmail.com")
        return "new@gmail.com"

    monkeypatch.setattr(
        "ai_employee.integrations.google_oauth.complete_authorization", fake_complete
    )
    me = (await client.get("/ui/api/me")).json()
    assert me["kind"] == "guest"

    page = await client.get("/ui/api/google/oauth/callback?code=c&state=s")
    assert page.status_code == 200 and "signed in as new@gmail.com" in page.text
    me = (await client.get("/ui/api/me")).json()
    assert me["kind"] == "account" and me["email"] == "new@gmail.com" and me["name"] == "Me"


async def test_google_connect_keeps_an_existing_account_signed_in(settings, client, monkeypatch):
    from ai_employee.api.users import COOKIE_NAME, encode_cookie, upsert_google_user

    _client_file(settings)
    owner = upsert_google_user(settings, {"email": "owner@x.com", "name": "Owner"})
    client.cookies.set(COOKIE_NAME, encode_cookie(settings, owner))

    async def fake_complete(s, state, code):
        _account(s, "other@gmail.com")
        return "other@gmail.com"

    monkeypatch.setattr(
        "ai_employee.integrations.google_oauth.complete_authorization", fake_complete
    )
    page = await client.get("/ui/api/google/oauth/callback?code=c&state=s")
    assert "other@gmail.com connected" in page.text and "signed in as" not in page.text
    assert (await client.get("/ui/api/me")).json()["email"] == "owner@x.com"


async def test_google_card_test_reports_disabled_api(settings, monkeypatch):
    """Scopes granted but the API is off in the Cloud project → a clear error."""
    import httpx

    from ai_employee.tools.builtins.google_workspace import SERVICE_PROBES

    _client_file(settings)
    _account(settings)
    seen: list[str] = []
    real_client = httpx.AsyncClient
    body = {
        "error": {
            "code": 403,
            "message": "Google Calendar API has not been used in project 42 before or it is "
            "disabled. Enable it by visiting https://console.developers.google.com/x",
            "status": "PERMISSION_DENIED",
        }
    }

    def fake_client(**kwargs):
        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(403, json=body)

        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    result = await connect_server(settings, "google_calendar")
    assert not result["ok"]
    assert "not enabled in your Google Cloud project" in result["error"]
    assert "https://console.developers.google.com/x" in result["error"]
    assert seen == [SERVICE_PROBES["google_calendar"]]
