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
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import IO, Any

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


def _leaf_errors(exc: BaseException) -> list[BaseException]:
    """The real exceptions inside (possibly nested) exception groups."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaf_errors(sub)]
    return [exc]


def describe_error(exc: BaseException, server_output: str = "") -> str:
    """A one-line, human-readable reason for a failed MCP connection or call.

    The SDK wraps failures in ``ExceptionGroup``s and reports a crashed stdio
    server only as "Connection closed", so the useful part is usually the
    leaf exception plus whatever the server printed to stderr before dying.
    """
    if isinstance(exc, ToolError):
        return str(exc)
    parts = []
    for leaf in _leaf_errors(exc):
        text = str(leaf).strip()
        parts.append(f"{type(leaf).__name__}: {text}" if text else type(leaf).__name__)
    message = "; ".join(dict.fromkeys(parts)) or type(exc).__name__
    if isinstance(exc, FileNotFoundError) or any(
        isinstance(leaf, FileNotFoundError) for leaf in _leaf_errors(exc)
    ):
        message += " (is the server's launcher — npx/uvx — installed and on PATH?)"
    output = server_output.strip()
    if output:
        message += f" — server said: {output}"
    return message


def _stderr_tail(errlog: IO[str] | None, limit: int = 600) -> str:
    """The last meaningful lines a stdio server wrote to stderr."""
    if errlog is None:
        return ""
    try:
        errlog.flush()
        errlog.seek(0)
        lines = [ln.strip() for ln in errlog.read().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return ""
    # Package-manager chatter ("Installed 31 packages…", npm warnings) is noise.
    noise = ("Installed ", "Resolved ", "Downloading ", "npm WARN", "npm notice")
    useful = [ln for ln in lines if not ln.startswith(noise)]
    return " | ".join(useful)[-limit:]


@asynccontextmanager
async def _session(server: RenderedServer, *, sse: bool = False, errlog: IO[str] | None = None):
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
        stdio = stdio_client(params, errlog=errlog) if errlog is not None else stdio_client(params)
        async with (
            stdio as (read, write),
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
    # Keep a stdio server's stderr: when it exits early, its own message ("OAuth
    # keys file not found…") is the only useful explanation of the failure.
    errlog = (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")  # noqa: SIM115
        if server.transport == "stdio"
        else None
    )

    async def _attempt(sse: bool) -> Any:
        async with _session(server, sse=sse, errlog=errlog) as session:
            return await fn(session)

    try:
        try:
            return await asyncio.wait_for(_attempt(False), timeout=timeout)
        except (TimeoutError, ToolError):
            raise
        except Exception as exc:  # noqa: BLE001 - retry older HTTP servers over SSE
            if server.transport != "http":
                raise
            log.debug("mcp.http_fallback_sse", server=server.id, error=str(exc))
            return await asyncio.wait_for(_attempt(True), timeout=timeout)
    except ToolError:
        raise
    except TimeoutError as exc:
        hint = _stderr_tail(errlog)
        raise ToolError(
            f"no response from MCP server {server.id!r} within {timeout:.0f}s"
            + (f" — server said: {hint}" if hint else "")
        ) from exc
    except Exception as exc:  # noqa: BLE001 - one readable reason for UI and agents
        raise ToolError(describe_error(exc, _stderr_tail(errlog))) from exc
    finally:
        if errlog is not None:
            errlog.close()


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
    "describe_error",
    "probe_server",
    "runtime_available",
    "sdk_available",
]
