"""Resource-lifecycle audit log.

Every state change to the resource catalog — create, dedup-reuse, promote,
delete, and denied access — is appended as one JSON line to the same
append-only ``audit_log_file`` used for sensitive tools, tagged
``"kind": "resource"``. One place then answers "what did this thing put on disk,
share, or try to touch?" across every task.

Auditing must never break a run, so every failure here is swallowed and logged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("resources.audit")


def record_resource_event(
    path: str | Path | None,
    *,
    action: str,
    resource_id: str = "",
    name: str = "",
    visibility: str = "",
    task: str | None = None,
    actor: str | None = None,
    ok: bool = True,
    detail: str = "",
) -> None:
    """Append one resource event. Never raises."""
    if path is None:
        return
    try:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": time.time(),
            "kind": "resource",
            "action": action,
            "resource_id": resource_id,
            "name": name,
            "visibility": visibility,
            "task": task,
            "actor": actor,
            "ok": ok,
            "detail": detail,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 - audit can't break a run
        log.warning("resource.audit_write_failed", error=str(exc))


__all__ = ["record_resource_event"]
