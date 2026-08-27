"""MCP proxy tool — exposes one remote MCP tool through the normal Tool API.

Each tool a connected MCP server advertises becomes an :class:`MCPProxyTool`
named ``mcp_<server>_<tool>``, so agents pick it up exactly like a builtin. Args
are passed through unvalidated (the server owns the real schema, which we
advertise verbatim to the LLM via ``to_spec``) and the call is executed over a
short-lived MCP session.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict

from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.mcp.client import RenderedServer, call_server_tool


class _PassthroughArgs(BaseModel):
    """The MCP server validates for real — accept whatever the LLM sends."""

    model_config = ConfigDict(extra="allow")


def proxy_tool_name(server_id: str, tool_name: str) -> str:
    raw = f"mcp_{server_id}_{tool_name}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)[:64]


class MCPProxyTool(Tool):
    category = "mcp"
    Args = _PassthroughArgs

    def __init__(
        self,
        server: RenderedServer,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any] | None,
        *,
        require_approval: bool = False,
        timeout: float = 120,
        token_provider: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        self._server = server
        self._tool_name = tool_name
        # OAuth connectors: called before every run so a refreshed access token
        # replaces the one baked into the frozen RenderedServer at build time.
        self._token_provider = token_provider
        self._schema = input_schema or {"type": "object", "properties": {}}
        self._timeout = timeout
        # name/description/approval_key are ClassVars on Tool; shadow them per
        # instance since every proxy is unique (setattr keeps mypy happy).
        setattr(self, "name", proxy_tool_name(server.id, tool_name))  # noqa: B010
        setattr(  # noqa: B010
            self,
            "description",
            f"[MCP:{server.id}] {description or tool_name}",
        )
        if require_approval:
            setattr(self, "approval_key", "mcp_tool")  # noqa: B010

    def to_spec(self):  # noqa: ANN201 - matches Tool.to_spec
        from ai_employee.brains.schemas import ToolSpec

        return ToolSpec(name=self.name, description=self.description, parameters=self._schema)

    async def _run(self, args: BaseModel) -> ToolResult:
        payload = args.model_dump(exclude_none=True)
        server = self._server
        if self._token_provider is not None:
            token = await self._token_provider()
            if token:
                server = replace(
                    server, headers={**server.headers, "Authorization": f"Bearer {token}"}
                )
        result = await call_server_tool(server, self._tool_name, payload, timeout=self._timeout)
        return ToolResult(
            ok=result["ok"],
            output=result["output"],
            error=None if result["ok"] else str(result["output"])[:500],
            metadata={"mcp_server": self._server.id, "mcp_tool": self._tool_name},
        )


__all__ = ["MCPProxyTool", "proxy_tool_name"]
