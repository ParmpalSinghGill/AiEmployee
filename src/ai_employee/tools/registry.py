"""Tool registry.

Holds the catalog of available tools, exposes them to brains as specs, and routes
execution through the human-approval policy.
"""

from __future__ import annotations

from collections.abc import Iterable

from ai_employee.brains.schemas import ToolSpec
from ai_employee.config import Settings, get_settings
from ai_employee.core.exceptions import ApprovalRequired
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip, clip_obj
from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.invocation_history import (
    current_tool_history,
    duplicate_payload,
)

log = get_logger("tool.registry")


class ToolRegistry:
    """A mutable collection of tools, addressable by name."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._settings = settings or get_settings()

    # ── Registration ─────────────────────────────────────────────────────────
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if tool.name in self._tools and not replace:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        log.debug("tool.register", tool=tool.name, category=tool.category)
        return tool

    def register_many(self, tools: Iterable[Tool], *, replace: bool = False) -> None:
        for t in tools:
            self.register(t, replace=replace)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── Lookup ───────────────────────────────────────────────────────────────
    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        """Tool specs to hand to a Brain. Optionally filter to a subset."""
        tools = (
            [self._tools[n] for n in names if n in self._tools]
            if names is not None
            else self._tools.values()
        )
        return [t.to_spec() for t in tools]

    # ── Execution ────────────────────────────────────────────────────────────
    async def execute(
        self,
        name: str,
        arguments: dict | None = None,
        *,
        approved: bool = False,
    ) -> ToolResult:
        """Run a tool by name, enforcing the approval policy.

        If the tool's ``approval_key`` is in ``settings.approval_required`` and the
        caller has not passed ``approved=True``, :class:`ApprovalRequired` is raised
        so the orchestrator can pause for a human.
        """
        tool = self.get(name)
        arguments = arguments or {}

        approval_key = tool.requires_approval(arguments)
        approved_by = "human" if approved else "auto"
        if approval_key and approval_key in self._settings.approval_required and not approved:
            from ai_employee.config.approval_policies import auto_allows

            if auto_allows(self._settings, approval_key, arguments):
                # Operator policy grants this call (recipients matched, quota
                # free) — proceed without pausing, and audit it as such.
                approved_by = "policy"
                log.info(
                    "tool.approval_policy_granted",
                    tool=name,
                    key=approval_key,
                    args=clip_obj(arguments),
                )
            else:
                log.info(
                    "tool.approval_required",
                    tool=name,
                    key=approval_key,
                    args=clip_obj(arguments),
                )
                raise ApprovalRequired(approval_key, payload={"tool": name, "args": arguments})

        history = current_tool_history()
        if history is not None:
            previous = history.get(name, arguments)
            if previous is not None and previous.result.ok:
                payload = duplicate_payload(previous.result, name)
                log.info(
                    "tool.duplicate_blocked",
                    tool=name,
                    args=clip_obj(arguments),
                    reason="Identical tool call already executed.",
                    first_at=previous.timestamp,
                )
                return ToolResult(
                    ok=True,
                    output=payload,
                    metadata={
                        "duplicate": True,
                        "first_at": previous.timestamp,
                        "tool": name,
                        "arguments": arguments,
                    },
                )

        log.info("tool.execute", tool=name, args=clip_obj(arguments))
        try:
            result = await tool.execute(**arguments)
        except ApprovalRequired:
            raise
        log.info(
            "tool.result",
            tool=name,
            ok=result.ok,
            latency_ms=result.latency_ms,
            output=clip(result.as_text()),
            error=result.error,
        )
        if approval_key:  # sensitive op executed — put it on the audit trail
            import structlog

            from ai_employee.observability.audit import record_action

            record_action(
                self._settings,
                tool=name,
                approval_key=approval_key,
                arguments=arguments,
                ok=result.ok,
                approved_by=approved_by,
                task_id=structlog.contextvars.get_contextvars().get("task_id"),
            )
        if history is not None and result.ok:
            history.record(name, arguments, result)
        return result


def default_registry(settings: Settings | None = None) -> ToolRegistry:
    """Build a registry pre-loaded with all built-in tools + enabled MCP connectors."""
    from ai_employee.tools.builtins import build_builtin_tools
    from ai_employee.tools.mcp import build_mcp_tools

    s = settings or get_settings()
    reg = ToolRegistry(s)
    reg.register_many(build_builtin_tools(s))
    try:
        # Offline-safe: proxies come from the cached tool lists in the MCP store;
        # no server is spawned or contacted until a tool actually runs.
        reg.register_many(build_mcp_tools(s), replace=True)
    except Exception as exc:  # noqa: BLE001 - a bad MCP config must not block startup
        log.warning("mcp.load_failed", error=str(exc))
    return reg
