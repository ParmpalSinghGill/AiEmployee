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
from ai_employee.tools.mcp.client import RenderedServer, probe_server
from ai_employee.tools.mcp.proxy import MCPProxyTool
from ai_employee.tools.mcp.store import load_servers, save_server

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


def build_mcp_tools(settings: Settings) -> list[Tool]:
    """Proxy tools for every enabled server, from the cached tool lists only."""
    tools: list[Tool] = []
    for server_id, entry in load_servers(settings).items():
        if not entry.get("enabled"):
            continue
        try:
            spec = connector_for(entry.get("connector", server_id))
            auth = entry.get("auth") or {}
            rendered = render_server(
                spec, entry.get("params") or {}, access_token=auth.get("access_token")
            )
        except (KeyError, ValueError) as exc:
            log.warning("mcp.build_skipped", server=server_id, error=str(exc))
            continue
        # OAuth connectors fetch a fresh (possibly refreshed) token per call —
        # the frozen RenderedServer would otherwise go stale when tokens rotate.
        token_provider = (
            _make_token_provider(settings, server_id) if spec.auth == "oauth" else None
        )
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
                )
            )
    return tools


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
        message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
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


__all__ = ["build_mcp_tools", "connect_server", "render_server", "sync_registry"]
