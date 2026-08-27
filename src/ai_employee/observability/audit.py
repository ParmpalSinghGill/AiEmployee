"""Cross-task audit trail of outward-facing actions.

Every execution of a *sensitive* tool (anything with an ``approval_key`` — mail
sends, calendar writes, shell writes, downloads, MCP calls, …) is appended to a
JSONL ledger, whether it was human-approved, policy-approved, or auto-approved.
One place answers "what has this thing actually done in the world?" across all
tasks and sessions. The approval-policy daily quotas also count from here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("audit")

_MAX_ARG_CHARS = 400


def _path(settings: Any) -> Path:
    return Path(settings.audit_log_file)


def _clip_args(arguments: dict[str, Any]) -> dict[str, Any]:
    clipped: dict[str, Any] = {}
    for k, v in arguments.items():
        s = v if isinstance(v, (int, float, bool)) or v is None else str(v)
        if isinstance(s, str) and len(s) > _MAX_ARG_CHARS:
            s = s[:_MAX_ARG_CHARS] + "…"
        clipped[k] = s
    return clipped


def record_action(
    settings: Any,
    *,
    tool: str,
    approval_key: str,
    arguments: dict[str, Any],
    ok: bool,
    approved_by: str,  # "human" | "policy" | "auto" | "pre"
    task_id: str | None = None,
) -> None:
    """Append one outward action. Must never raise — auditing can't break a run."""
    try:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": time.time(),
            "task_id": task_id,
            "tool": tool,
            "key": approval_key,
            "args": _clip_args(arguments or {}),
            "ok": ok,
            "approved_by": approved_by,
        }
        p = _path(settings)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("audit.write_failed", error=str(exc))


def read_actions(settings: Any, *, limit: int = 200, key: str | None = None) -> list[dict]:
    """Most-recent-first slice of the trail (optionally one approval key)."""
    p = _path(settings)
    if not p.exists():
        return []
    entries: list[dict] = []
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if key and e.get("key") != key:
                    continue
                entries.append(e)
    except OSError as exc:
        log.warning("audit.read_failed", error=str(exc))
        return []
    return list(reversed(entries[-limit:] if limit else entries))


def count_today(settings: Any, key: str) -> int:
    """How many actions with this approval key happened today (local time)."""
    today = time.strftime("%Y-%m-%d")
    return sum(
        1
        for e in read_actions(settings, limit=0, key=key)
        if str(e.get("ts", "")).startswith(today) and e.get("ok")
    )


__all__ = ["count_today", "read_actions", "record_action"]
