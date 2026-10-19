"""Exception types for the resource-management subsystem.

All subclass :class:`~ai_employee.core.exceptions.ToolError` so a resource
operation that fails (a path escape, a denied cross-task read, a missing
resource) surfaces to the agent loop as a clean ``ToolResult(ok=False)`` instead
of an unhandled traceback.
"""

from __future__ import annotations

from typing import Any

from ai_employee.core.exceptions import ToolError


class ResourceError(ToolError):
    """Base class for every resource-layer failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, tool="resources")
        self.details = details or {}


class ResourcePathError(ResourceError):
    """A path escaped the workspace / task sandbox, or a name was unsafe."""


class ResourceAccessDenied(ResourceError):
    """An actor tried to read or mutate a resource it is not authorized to touch."""


class ResourceNotFound(ResourceError):
    """No resource matched the given id, name or hash."""


class ResourceConflict(ResourceError):
    """A concurrent mutation lost a race or violated an invariant."""


__all__ = [
    "ResourceError",
    "ResourcePathError",
    "ResourceAccessDenied",
    "ResourceNotFound",
    "ResourceConflict",
]
