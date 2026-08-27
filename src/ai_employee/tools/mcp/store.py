"""Persistence for configured MCP servers.

Mirrors :mod:`ai_employee.config.secrets_store`: a JSON file
(``settings.mcp_config_file``) managed from the UI. Besides the connection
parameters it caches each server's discovered tool list, so the registry can
expose MCP tools at startup without spawning/contacting any server (offline
startup stays fast and dependency-free); the live connection happens only when
a tool is actually called or the connector is (re)tested.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("mcp.store")


def _path(settings: Any) -> Path:
    return Path(settings.mcp_config_file)


def load_servers(settings: Any) -> dict[str, dict[str, Any]]:
    """All configured servers, keyed by connector id."""
    p = _path(settings)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        servers = data.get("servers") if isinstance(data, dict) else None
        return servers if isinstance(servers, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a corrupt store must not crash startup
        log.warning("mcp.store.load_failed", error=str(exc))
        return {}


def _save(settings: Any, servers: dict[str, dict[str, Any]]) -> None:
    p = _path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"servers": servers}, indent=2))
    with contextlib.suppress(OSError):  # params may hold tokens — keep owner-only
        p.chmod(0o600)


def save_server(
    settings: Any,
    server_id: str,
    *,
    params: dict[str, str] | None = None,
    enabled: bool | None = None,
    require_approval: bool | None = None,
    tools: list[dict[str, Any]] | None = None,
    error: str | None = None,
    connected_at: str | None = None,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or update one server entry; only the passed fields change."""
    servers = load_servers(settings)
    entry = servers.get(server_id) or {
        "connector": server_id,
        "enabled": True,
        "params": {},
        "require_approval": False,
        "tools": [],
    }
    if params is not None:
        entry["params"] = params
    if enabled is not None:
        entry["enabled"] = enabled
    if require_approval is not None:
        entry["require_approval"] = require_approval
    if tools is not None:
        entry["tools"] = tools
        entry["error"] = None
    if error is not None:
        entry["error"] = error
    if connected_at is not None:
        entry["connected_at"] = connected_at
    if auth is not None:
        entry["auth"] = auth
    servers[server_id] = entry
    _save(settings, servers)
    log.info("mcp.store.saved", server=server_id, enabled=entry["enabled"])
    return entry


def delete_server(settings: Any, server_id: str) -> bool:
    servers = load_servers(settings)
    if server_id not in servers:
        return False
    servers.pop(server_id)
    _save(settings, servers)
    log.info("mcp.store.deleted", server=server_id)
    return True


__all__ = ["load_servers", "save_server", "delete_server"]
