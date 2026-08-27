"""Thin async client over the ``mcp`` SDK.

Sessions are short-lived: each probe/call opens a connection, does its work, and
closes. That trades a little latency (stdio servers are respawned per call) for
robustness — no background session tasks to babysit across FastAPI requests,
employee rebuilds, or server crashes. Discovered tool lists are cached in the
MCP store, so sessions are only opened when a tool actually runs.

The ``mcp`` SDK is imported lazily: the framework (and the offline test suite)
works without it; connecting simply reports that the package is missing.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ai_employee.core.exceptions import ToolError
from ai_employee.core.logging import get_logger

log = get_logger("mcp.client")


@dataclass(frozen=True)
class RenderedServer:
    """A connector spec with all ``{param}`` templates resolved — ready to launch."""

    id: str
    transport: str  # "stdio" | "http"
    command: str = ""
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


def sdk_available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


@asynccontextmanager
async def _session(server: RenderedServer, *, sse: bool = False):
    """Open an initialized ClientSession for the server (stdio or HTTP)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import get_default_environment, stdio_client

    if server.transport == "stdio":
        argv = shlex.split(server.command)
        if not argv:
            raise ToolError(f"mcp server {server.id!r} has an empty command")
        params = StdioServerParameters(
            command=argv[0],
            args=argv[1:],
            env={**get_default_environment(), **server.env},
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session
    elif server.transport == "http":
        headers = {k: v for k, v in server.headers.items() if v}
        if sse:  # legacy servers that only speak HTTP+SSE
            from mcp.client.sse import sse_client

            async with (
                sse_client(server.url, headers=headers or None) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
        else:
            from mcp.client.streamable_http import streamablehttp_client

            async with (
                streamablehttp_client(server.url, headers=headers or None) as (
                    read,
                    write,
                    _get_session_id,
                ),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
    else:
        raise ToolError(f"unknown mcp transport: {server.transport!r}")


async def _with_session(
    server: RenderedServer,
    fn: Callable[[Any], Awaitable[Any]],
    *,
    timeout: float,
) -> Any:
    """Run ``fn(session)`` with a timeout; HTTP servers fall back to SSE once."""
    if not sdk_available():
        raise ToolError("the 'mcp' package is not installed — run: pip install mcp")

    async def _attempt(sse: bool) -> Any:
        async with _session(server, sse=sse) as session:
            return await fn(session)

    try:
        return await asyncio.wait_for(_attempt(False), timeout=timeout)
    except (TimeoutError, ToolError):
        raise
    except Exception as exc:  # noqa: BLE001 - retry older HTTP servers over SSE
        if server.transport != "http":
            raise
        log.debug("mcp.http_fallback_sse", server=server.id, error=str(exc))
        return await asyncio.wait_for(_attempt(True), timeout=timeout)


async def probe_server(server: RenderedServer, *, timeout: float = 120) -> list[dict[str, Any]]:
    """Connect, list the server's tools, and return them as plain dicts.

    The generous default timeout covers ``npx``/``uvx`` downloading the server
    package on first use.
    """

    async def _list(session: Any) -> list[dict[str, Any]]:
        listed = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            }
            for t in listed.tools
        ]

    tools: list[dict[str, Any]] = await _with_session(server, _list, timeout=timeout)
    log.info("mcp.probed", server=server.id, tools=len(tools))
    return tools


async def call_server_tool(
    server: RenderedServer,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    """Call one tool and flatten the MCP result to ``{ok, output}``."""

    async def _call(session: Any) -> dict[str, Any]:
        result = await session.call_tool(tool_name, arguments or {})
        return _flatten_result(result)

    out: dict[str, Any] = await _with_session(server, _call, timeout=timeout)
    return out


def _flatten_result(result: Any) -> dict[str, Any]:
    """Reduce a CallToolResult to something an agent loop can consume as text."""
    if getattr(result, "structuredContent", None):
        return {"ok": not result.isError, "output": result.structuredContent}
    parts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        elif getattr(block, "type", "") == "image":
            parts.append(f"[image {getattr(block, 'mimeType', 'unknown')}]")
        else:
            parts.append(str(block))
    return {"ok": not result.isError, "output": "\n".join(parts)}


def runtime_available() -> dict[str, bool]:
    """Which MCP runtimes are usable on this machine (for the UI's hints)."""
    import shutil

    return {
        "sdk": sdk_available(),
        "npx": shutil.which("npx") is not None,
        "uvx": shutil.which("uvx") is not None,
    }


__all__ = [
    "RenderedServer",
    "call_server_tool",
    "probe_server",
    "runtime_available",
    "sdk_available",
]
