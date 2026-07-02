"""Filesystem sandbox helper.

All file/code tools resolve paths through :func:`resolve_in_workspace`, which keeps
operations confined to the configured workspace directory and blocks traversal
outside it. This is the framework's first line of defense for autonomous file ops.
"""

from __future__ import annotations

from pathlib import Path

from ai_employee.core.exceptions import ToolError


def ensure_workspace(workspace: Path) -> Path:
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def resolve_in_workspace(workspace: Path, relative: str, *, must_exist: bool = False) -> Path:
    """Resolve ``relative`` under ``workspace``; raise if it escapes the sandbox."""
    workspace = ensure_workspace(workspace)
    candidate = (workspace / relative).resolve()
    if workspace != candidate and workspace not in candidate.parents:
        raise ToolError(
            f"path {relative!r} escapes the workspace sandbox",
            tool="filesystem",
        )
    if must_exist and not candidate.exists():
        raise ToolError(f"path does not exist: {relative!r}", tool="filesystem")
    return candidate
