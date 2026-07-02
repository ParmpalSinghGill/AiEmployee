"""Context for human resource decisions during a paused task resume."""

from __future__ import annotations

import contextvars
from typing import Any

_decision_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "resource_decision", default=None
)


def bind_resource_decision(decision: dict[str, Any] | None) -> None:
    _decision_var.set(decision)


def get_resource_decision() -> dict[str, Any] | None:
    return _decision_var.get()


def clear_resource_decision() -> None:
    _decision_var.set(None)


__all__ = ["bind_resource_decision", "clear_resource_decision", "get_resource_decision"]
