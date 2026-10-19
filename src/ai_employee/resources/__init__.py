"""Secure, task-isolated resource management.

Separates the five concepts the design calls out — task identity, resource
identity, ownership, visibility and physical storage — so multiple tasks can
share a workspace without mixing their private data:

    from ai_employee.resources import ResourceManager

    mgr = ResourceManager(settings.workspace_dir, audit_file=settings.audit_log_file)
    rec = mgr.put(model_path, name="yolo26n.pt", visibility="shared", kind="model")
    mgr.authorize(rec, task=some_task_key, mode="read")   # raises if not allowed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_employee.resources.errors import (
    ResourceAccessDenied,
    ResourceConflict,
    ResourceError,
    ResourceNotFound,
    ResourcePathError,
)
from ai_employee.resources.migration import migrate_workspace
from ai_employee.resources.models import (
    Provenance,
    ResourceLink,
    ResourceRecord,
    ResourceVersion,
    Visibility,
)
from ai_employee.resources.store import ResourceManager

__all__ = [
    "ResourceManager",
    "ResourceRecord",
    "ResourceVersion",
    "ResourceLink",
    "Provenance",
    "Visibility",
    "ResourceError",
    "ResourcePathError",
    "ResourceAccessDenied",
    "ResourceNotFound",
    "ResourceConflict",
    "migrate_workspace",
    "manager_for",
]


def manager_for(settings: Any, *, workspace: str | Path | None = None) -> ResourceManager:
    """Build a :class:`ResourceManager` wired to the app's settings (audit file)."""
    ws = Path(workspace) if workspace is not None else Path(settings.workspace_dir)
    audit = getattr(settings, "audit_log_file", None)
    return ResourceManager(ws, audit_file=audit)
