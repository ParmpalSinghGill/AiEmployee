"""Per-task tool invocation history for duplicate-call prevention.

The orchestrator opens a history scope for each task run. :class:`ToolRegistry`
checks it before executing a tool so identical (tool, args) pairs are never run
twice in the same task.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_employee.tools.base import ToolResult

_history_var: contextvars.ContextVar[TaskToolHistory | None] = contextvars.ContextVar(
    "task_tool_history", default=None
)

# Dedup is scoped to one agent's turn, not the whole task. Task-wide dedup meant a
# *read* another agent already made (list_dir on the dataset) was permanently
# blocked for everyone else — so a later step that legitimately needed to inspect
# the same directory got a "duplicate" wall, tripped the repeated-duplicate abort,
# and gave up before it could run its real work.
_scope_var: contextvars.ContextVar[str] = contextvars.ContextVar("tool_dedup_scope", default="")


def current_dedup_scope() -> str:
    return _scope_var.get()


def bind_dedup_scope(scope: str) -> contextvars.Token:
    """Scope subsequent duplicate checks (e.g. to one agent's turn)."""
    return _scope_var.set(scope or "")


def reset_dedup_scope(token: contextvars.Token) -> None:
    with contextlib.suppress(ValueError):
        _scope_var.reset(token)


def normalize_tool_arguments(arguments: dict[str, Any] | None) -> str:
    """Stable string form of tool arguments (key order must not matter)."""
    try:
        return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - never let dedup keying break execution
        return str(arguments)


#: What to do instead, per tool family. A bare "duplicate" tells the model what not
#: to do but not what to do next, so it retries a near-identical call and burns the
#: iteration budget. Naming the right next tool breaks that loop.
_NEXT_STEP_HINT: dict[str, str] = {
    "list_dir": (
        "You already have this listing. To count, measure or parse files, call "
        "python_exec once and compute it — do not list again. To see a whole tree, "
        "call list_dir with recursive=true."
    ),
    "read_file": ("You already have this file's contents. Use them, or read a different path."),
    "web_search": "Use the results you have, or search a clearly different phrase.",
    "hf_search": (
        "Use the hits you have. For other modalities try kaggle_search, "
        "roboflow_search or hub_search_all, or move on to download_resource."
    ),
    "hub_search_all": (
        "Use the hits you have, or query a specific hub (hf_search, kaggle_search, "
        "roboflow_search) with a different term."
    ),
}


def duplicate_payload(previous: ToolResult, tool: str = "") -> dict[str, Any]:
    hint = _NEXT_STEP_HINT.get(
        tool, "Reuse the previous result or make a materially different call."
    )
    return {
        "status": "duplicate",
        "message": f"This exact call already ran in this turn. {hint}",
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
        return f"{current_dedup_scope()}|{tool}:{normalize_tool_arguments(arguments)}"

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
    "bind_dedup_scope",
    "current_dedup_scope",
    "reset_dedup_scope",
    "clear_task_tool_history",
    "current_tool_history",
    "duplicate_payload",
    "normalize_tool_arguments",
]
