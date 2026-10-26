"""Glue between the connector catalog, the store, and the tool registry.

``build_mcp_tools`` is the offline path: it turns cached tool lists into proxy
tools with zero network/subprocess activity, so it is safe to call while
building the default registry at startup. ``connect_server`` is the online path
used by the UI: render the connector, probe it, and cache what it advertises.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_employee.config import Settings
from ai_employee.core.logging import get_logger
from ai_employee.tools.base import Tool
from ai_employee.tools.mcp.catalog import ConnectorSpec, connector_for
from ai_employee.tools.mcp.client import RenderedServer, describe_error, probe_server
from ai_employee.tools.mcp.proxy import MCPProxyTool
from ai_employee.tools.mcp.store import delete_server, load_servers, save_server

log = get_logger("mcp.manager")


def render_server(
    spec: ConnectorSpec, params: dict[str, str], *, access_token: str | None = None
) -> RenderedServer:
    """Resolve a connector's ``{param}`` templates against saved params."""
    values = {p.name: (params.get(p.name) or p.default) for p in spec.params}
    missing = [p.label for p in spec.params if p.required and not values.get(p.name, "").strip()]
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}")

    def fill(template: str) -> str:
        return template.format(**values)

    # Drop headers whose params were left blank (e.g. optional bearer token).
    headers = {
        k: fill(v)
        for k, v in spec.headers.items()
        if all(values.get(p.name) for p in spec.params if "{" + p.name + "}" in v)
    }
    if access_token:  # OAuth connectors: the browser-approved grant is the auth
        headers["Authorization"] = f"Bearer {access_token}"
    return RenderedServer(
        id=spec.id,
        transport=spec.transport,
        command=fill(spec.command) if spec.command else "",
        env={k: fill(v) for k, v in spec.env.items()},
        url=fill(spec.url) if spec.url else "",
        headers=headers,
    )


def google_connector_state(settings: Settings, spec: ConnectorSpec) -> dict[str, Any]:
    """Live status of a Google-backed card, read from the connected accounts.

    ``ready`` means some connected account granted every scope the card needs,
    so its native tools work right now. ``missing_scopes`` lists what a
    re-consent would add when accounts exist but none has the full set.
    """
    from ai_employee.integrations.google_oauth import client_config, load_accounts

    configured = client_config(settings) is not None
    accounts = load_accounts(settings) if configured else {}
    ready_accounts: list[str] = []
    missing: set[str] = set()
    for email, entry in sorted(accounts.items()):
        granted = set(str(entry.get("scopes") or "").split())
        lacking = [scope for scope in spec.google_scopes if scope not in granted]
        if lacking:
            missing.update(lacking)
        else:
            ready_accounts.append(email)
    return {
        "configured": configured,
        "ready": bool(ready_accounts),
        "accounts": ready_accounts,
        "connected_accounts": sorted(accounts),
        "missing_scopes": [] if ready_accounts else sorted(missing),
        "tools": list(spec.native_tools) if ready_accounts else [],
    }


async def probe_google_service(settings: Settings, server_id: str, account: str) -> str | None:
    """Live check of a Google-backed card (patched out in offline tests)."""
    from ai_employee.tools.builtins.google_workspace import probe_service

    return await probe_service(settings, server_id, account)


def build_mcp_tools(settings: Settings) -> list[Tool]:
    """Proxy tools for every enabled server, from the cached tool lists only."""
    tools: list[Tool] = []
    for server_id, entry in load_servers(settings).items():
        if not entry.get("enabled"):
            continue
        try:
            spec = connector_for(entry.get("connector", server_id))
            if spec.auth == "google":
                continue  # served by the native Google tools, not a proxy
            auth = entry.get("auth") or {}
            rendered = render_server(
                spec, entry.get("params") or {}, access_token=auth.get("access_token")
            )
        except (KeyError, ValueError) as exc:
            log.warning("mcp.build_skipped", server=server_id, error=str(exc))
            continue
        # OAuth connectors fetch a fresh (possibly refreshed) token per call —
        # the frozen RenderedServer would otherwise go stale when tokens rotate.
        token_provider = _make_token_provider(settings, server_id) if spec.auth == "oauth" else None
        for t in entry.get("tools") or []:
            tools.append(
                MCPProxyTool(
                    rendered,
                    t["name"],
                    t.get("description", ""),
                    t.get("input_schema"),
                    require_approval=bool(entry.get("require_approval")),
                    timeout=settings.mcp_call_timeout,
                    token_provider=token_provider,
                    agents=spec.agents,
                    server_label=spec.label,
                )
            )
    return tools


def connected_services_by_agent(settings: Settings) -> dict[str, list[str]]:
    """Agent name → labels of connected MCP servers whose tools it receives.

    ``"*"`` collects servers routed to every worker. Read from the store only
    (no network), so it is cheap enough to call on every roster build.
    """
    out: dict[str, list[str]] = {}
    for server_id, entry in load_servers(settings).items():
        if not entry.get("enabled") or not entry.get("tools"):
            continue
        try:
            spec = connector_for(entry.get("connector", server_id))
        except KeyError:
            continue
        if spec.auth == "google":
            continue
        for agent in spec.agents or ("*",):
            out.setdefault(agent, []).append(spec.label)
    return out


def _make_token_provider(settings: Settings, server_id: str):  # noqa: ANN202
    async def provider() -> str | None:
        from ai_employee.tools.mcp.oauth import ensure_access_token

        return await ensure_access_token(settings, server_id)

    return provider


async def connect_server(
    settings: Settings,
    server_id: str,
    *,
    params: dict[str, str] | None = None,
    require_approval: bool | None = None,
) -> dict[str, Any]:
    """Save the server config, connect, discover its tools, and cache them.

    Failures are recorded on the entry (and re-raised as the return payload's
    ``error``) so the UI can show why a connector is down without a stack trace.
    """
    spec = connector_for(server_id)
    if spec.auth == "google":
        # Nothing to launch: the card is live as soon as a connected Google
        # account has granted its scopes. Otherwise the UI runs Google consent.
        state = google_connector_state(settings, spec)
        if server_id in load_servers(settings):  # drop a stale pre-Google-backed entry
            delete_server(settings, server_id)
        if state["ready"]:
            # A granted scope isn't enough: the API must also be enabled in the
            # operator's Google Cloud project. One cheap read call proves both.
            error = await probe_google_service(settings, server_id, state["accounts"][0])
            if error:
                return {"ok": False, "server": server_id, "error": error, "tools": []}
            return {"ok": True, "server": server_id, "error": None, "tools": state["tools"]}
        if not state["configured"]:
            return {
                "ok": False,
                "server": server_id,
                "tools": [],
                "error": (
                    "Google sign-in is not set up: put your OAuth client JSON at "
                    f"{settings.google_oauth_client_file} and restart."
                ),
            }
        return {
            "ok": False,
            "server": server_id,
            "tools": [],
            "needs_google": True,
            "missing_scopes": state["missing_scopes"],
        }
    entry = save_server(
        settings, server_id, params=params, require_approval=require_approval, enabled=True
    )
    try:
        access_token: str | None = None
        if spec.auth == "oauth":
            from ai_employee.tools.mcp.oauth import ensure_access_token

            access_token = await ensure_access_token(settings, server_id)
            if not access_token:
                # Never authorized: the UI must run the browser consent flow
                # first (see /ui/api/mcp/servers/{id}/authorize), then retry.
                return {"ok": False, "server": server_id, "needs_auth": True, "tools": []}
        rendered = render_server(spec, entry.get("params") or {}, access_token=access_token)
        tools = await probe_server(rendered, timeout=settings.mcp_call_timeout)
    except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
        message = describe_error(exc)
        save_server(settings, server_id, error=message)
        log.warning("mcp.connect_failed", server=server_id, error=message)
        return {"ok": False, "server": server_id, "error": message, "tools": []}
    save_server(
        settings,
        server_id,
        tools=tools,
        connected_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return {"ok": True, "server": server_id, "error": None, "tools": tools}


def sync_registry(settings: Settings, registry: Any) -> int:
    """Replace the registry's ``mcp_*`` tools with the current store contents.

    Called after any UI change so the live employee picks up connectors without
    a full rebuild. Returns the number of registered MCP tools.
    """
    for name in list(registry.names()):
        if name.startswith("mcp_"):
            registry.unregister(name)
    tools = build_mcp_tools(settings)
    registry.register_many(tools, replace=True)
    log.info("mcp.registry_synced", tools=len(tools))
    return len(tools)


__all__ = [
    "build_mcp_tools",
    "connect_server",
    "connected_services_by_agent",
    "google_connector_state",
    "render_server",
    "sync_registry",
]
