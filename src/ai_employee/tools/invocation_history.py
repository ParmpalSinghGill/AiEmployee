"""Per-task tool invocation history for duplicate-call prevention.

The orchestrator opens a history scope for each task run. :class:`ToolRegistry`
checks it before executing a tool so identical (tool, args) pairs are never run
twice in the same task.
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_employee.tools.base import ToolResult

_history_var: contextvars.ContextVar[TaskToolHistory | None] = contextvars.ContextVar(
    "task_tool_history", default=None
)


def normalize_tool_arguments(arguments: dict[str, Any] | None) -> str:
    """Stable string form of tool arguments (key order must not matter)."""
    try:
        return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - never let dedup keying break execution
        return str(arguments)


def duplicate_payload(previous: ToolResult) -> dict[str, Any]:
    return {
        "status": "duplicate",
        "message": (
            "This tool call has already been executed. Reuse the previous result "
            "or generate a different query."
        ),
        "previous_result": previous.output,
    }


@dataclass(frozen=True)
class ToolInvocationRecord:
    tool: str
    arguments: dict[str, Any]
    normalized_args: str
    timestamp: str
    result: ToolResult


class TaskToolHistory:
    """In-memory log of successful tool calls for one task run."""

    def __init__(self) -> None:
        self._records: dict[str, ToolInvocationRecord] = {}

    @staticmethod
    def signature(tool: str, arguments: dict[str, Any] | None) -> str:
        return f"{tool}:{normalize_tool_arguments(arguments)}"

    def get(self, tool: str, arguments: dict[str, Any] | None) -> ToolInvocationRecord | None:
        return self._records.get(self.signature(tool, arguments))

    def record(self, tool: str, arguments: dict[str, Any] | None, result: ToolResult) -> None:
        norm = normalize_tool_arguments(arguments)
        self._records[self.signature(tool, arguments)] = ToolInvocationRecord(
            tool=tool,
            arguments=dict(arguments or {}),
            normalized_args=norm,
            timestamp=datetime.now(UTC).isoformat(),
            result=result,
        )

    def entries(self) -> list[ToolInvocationRecord]:
        return list(self._records.values())


def begin_task_tool_history() -> TaskToolHistory:
    history = TaskToolHistory()
    _history_var.set(history)
    return history


def clear_task_tool_history() -> None:
    _history_var.set(None)


def current_tool_history() -> TaskToolHistory | None:
    return _history_var.get()


__all__ = [
    "TaskToolHistory",
    "ToolInvocationRecord",
    "begin_task_tool_history",
    "clear_task_tool_history",
    "current_tool_history",
    "duplicate_payload",
    "normalize_tool_arguments",
]
