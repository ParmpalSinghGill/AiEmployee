"""Agent tools for the secure resource catalog.

These expose the :mod:`ai_employee.resources` manager to agents as ordinary tools:
list what the current task may use, resolve a resource to a path, and promote a
task-private artifact to shared. Authorization is enforced by the manager, so a
task can never read another task's private files through these tools.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai_employee.config import Settings
from ai_employee.core.logging import get_logger
from ai_employee.observability.resource_ctx import get_resource_session, get_resource_task_id
from ai_employee.resources import ResourceManager
from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.builtins._sandbox import ensure_workspace

log = get_logger("resources.tools")


def _manager(settings: Settings) -> ResourceManager:
    ws = ensure_workspace(settings.workspace_dir)
    return ResourceManager(ws, audit_file=getattr(settings, "audit_log_file", None))


def _current_task(manager: ResourceManager) -> str | None:
    session = get_resource_session()
    return manager.task_identity(session) if session else None


def _current_ids(manager: ResourceManager) -> tuple[str, str, str]:
    """``(task_key, task_id, session_id)`` for the run currently executing."""
    session = get_resource_session()
    key = manager.task_identity(session) if session else ""
    return key, get_resource_task_id(), (session or "")


def _row(rec: Any, ws: Any) -> dict[str, Any]:
    from pathlib import Path

    return {
        "id": rec.id,
        "name": rec.name,
        "kind": rec.kind,
        "visibility": rec.visibility,
        "owner_task": rec.owner_task,
        "hash": rec.hash,
        "size": rec.size,
        "version": rec.version,
        "source": rec.source,
        "url": rec.url,
        "path": str(Path(ws) / rec.path),
        "relative_path": rec.path,
        "provenance": rec.provenance.to_dict(),
    }


class ListResourcesTool(Tool):
    name = "list_resources"
    description = (
        "List resources available to the CURRENT task: its own private files plus "
        "every shared, reusable resource in the workspace. Read-only and offline — "
        "no network, no approval. Use it to find a resource's id/path before "
        "get_resource, and to check what already exists before downloading."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        query: str = Field("", description="Free-text filter over name/kind/url/hash.")
        kind: str = Field("", description="Optional: model | dataset | repo | media | file")
        shared_only: bool = Field(False, description="Only shared, reusable resources.")

    async def _run(self, args: ListResourcesTool.Args) -> ToolResult:
        mgr = _manager(self._settings)
        task = _current_task(mgr)
        # A task sees its own private resources + all shared ones (enforced again
        # per-record by the manager's authorization).
        records = mgr.list(
            task=task,
            kind=args.kind.strip() or None,
            query=args.query,
            shared_only=args.shared_only,
        )
        items = []
        for r in records:
            row = _row(r, mgr.workspace)
            tasks = mgr.tasks_for(r.id)
            row["tasks"] = [t["task_key"] or t["task_id"] for t in tasks]
            row["shared_with_tasks"] = len(tasks)
            items.append(row)
        return ToolResult(
            ok=True,
            output={
                "task": task,
                "resources": items,
                "count": len(items),
                "note": (
                    "Private resources belong to this task; shared resources are "
                    "reusable by any task. 'tasks' lists every task linked to a source."
                    if task
                    else "No task bound — showing shared resources."
                ),
            },
            metadata={"count": len(items), "task": task},
        )


class GetResourceTool(Tool):
    name = "get_resource"
    description = (
        "Resolve a resource by id, name or content hash to a usable workspace path. "
        "Only resources this task is authorized to read are returned (its own "
        "private ones + shared ones). Read-only."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        ref: str = Field(description="Resource id, name (e.g. yolov8n.pt) or hash.")
        mark_used: bool = Field(True, description="Update the resource's last-used time.")

    async def _run(self, args: GetResourceTool.Args) -> ToolResult:
        mgr = _manager(self._settings)
        task = _current_task(mgr)
        record = mgr.resolve(args.ref, task=task, mode="read")
        if args.mark_used:
            mgr.set_last_used(record.id)
        # Record this task as a consumer of the source (task <-> source link), so
        # the catalog knows when several tasks share the same resource.
        key, tid, sid = _current_ids(mgr)
        if key or tid:
            mgr.link(record.id, task_key=key, task_id=tid, session_id=sid, role="consumer")
        return ToolResult(
            ok=True,
            output={"resource": _row(record, mgr.workspace)},
            metadata={"resource_id": record.id, "visibility": record.visibility},
        )


class PromoteResourceTool(Tool):
    name = "promote_resource"
    description = (
        "Promote one of this task's PRIVATE resources to SHARED, so every future "
        "task can reuse it. Requires approval (shell_write). Only the owning task "
        "may promote its own resource."
    )
    category = "research"
    approval_key = "shell_write"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        resource_id: str = Field(description="Id of the private resource to share.")

    async def _run(self, args: PromoteResourceTool.Args) -> ToolResult:
        mgr = _manager(self._settings)
        task = _current_task(mgr)
        record = mgr.promote(args.resource_id, task=task)
        return ToolResult(
            ok=True,
            output={
                "resource": _row(record, mgr.workspace),
                "note": "Resource is now shared and reusable by every task.",
            },
            metadata={"resource_id": record.id, "visibility": record.visibility},
        )


def build_resource_manager_tools(settings: Settings) -> list[Tool]:
    return [
        ListResourcesTool(settings),
        GetResourceTool(settings),
        PromoteResourceTool(settings),
    ]


__all__ = [
    "ListResourcesTool",
    "GetResourceTool",
    "PromoteResourceTool",
    "build_resource_manager_tools",
]
