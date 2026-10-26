"""Approvals scoped to one exact tool call ("this command only").

An operation approval (``shell_write``) lets every later call with that key run
for the rest of the task. A *command* approval lets only the exact call the
human saw run: same tool, same arguments. It is stored in the task's
``approved_ops`` as a ``call:`` token, so it travels through checkpoints,
resumes and persistence exactly like an operation key.

A run resumed after approval re-generates the paused step, so the token must
match a re-issued identical call; arguments that never change what runs
(``timeout``) are ignored. A different command asks again.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ai_employee.tools.invocation_history import normalize_tool_arguments

CALL_TOKEN_PREFIX = "call:"

#: Arguments that do not change what a call does, so a re-issued call that only
#: differs in them still counts as the approved command.
_IGNORED_ARGS = frozenset({"timeout"})


def approved_call_token(tool: str, arguments: dict[str, Any] | None) -> str:
    """The approval token for exactly this call."""
    args = {k: v for k, v in (arguments or {}).items() if k not in _IGNORED_ARGS}
    digest = hashlib.sha256(f"{tool}\n{normalize_tool_arguments(args)}".encode()).hexdigest()
    return f"{CALL_TOKEN_PREFIX}{tool}:{digest[:24]}"


def is_call_token(op: str) -> bool:
    return str(op).startswith(CALL_TOKEN_PREFIX)


def call_tokens_for_approval(approval: dict[str, Any]) -> set[str]:
    """Tokens for every call a pause was waiting on (the shown call + its batch)."""
    calls: list[dict[str, Any]] = []
    if approval.get("tool"):
        calls.append({"tool": approval.get("tool"), "args": approval.get("args")})
    calls.extend(c for c in (approval.get("batched_calls") or []) if c.get("tool"))
    return {
        approved_call_token(
            str(c["tool"]), c.get("args") if isinstance(c.get("args"), dict) else {}
        )
        for c in calls
    }


__all__ = [
    "CALL_TOKEN_PREFIX",
    "approved_call_token",
    "call_tokens_for_approval",
    "is_call_token",
]
