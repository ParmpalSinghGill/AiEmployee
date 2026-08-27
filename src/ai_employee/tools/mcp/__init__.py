"""MCP (Model Context Protocol) connectors.

A catalog of popular MCP servers (GitHub, Slack, Postgres, Playwright, …) that
can be enabled from the web console; each server's tools are proxied into the
normal :class:`~ai_employee.tools.registry.ToolRegistry` as ``mcp_*`` tools.
"""

from ai_employee.tools.mcp.catalog import CONNECTOR_CATALOG, ConnectorSpec, connector_for
from ai_employee.tools.mcp.client import RenderedServer, runtime_available, sdk_available
from ai_employee.tools.mcp.manager import (
    build_mcp_tools,
    connect_server,
    render_server,
    sync_registry,
)
from ai_employee.tools.mcp.proxy import MCPProxyTool, proxy_tool_name
from ai_employee.tools.mcp.store import delete_server, load_servers, save_server

__all__ = [
    "CONNECTOR_CATALOG",
    "ConnectorSpec",
    "MCPProxyTool",
    "RenderedServer",
    "build_mcp_tools",
    "connect_server",
    "connector_for",
    "delete_server",
    "load_servers",
    "proxy_tool_name",
    "render_server",
    "runtime_available",
    "save_server",
    "sdk_available",
    "sync_registry",
]
