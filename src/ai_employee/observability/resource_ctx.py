"""Context for human resource decisions and run-level approvals during resume."""

from __future__ import annotations

import contextvars
from typing import Any

_decision_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "resource_decision", default=None
)
_approved_ops_var: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "approved_ops", default=None
)
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "resource_session_id", default=None
)


def bind_resource_decision(decision: dict[str, Any] | None) -> None:
    _decision_var.set(decision)


def bind_approved_ops(ops: set[str] | frozenset[str] | None) -> None:
    _approved_ops_var.set(frozenset(ops or ()))


def bind_resource_session(session_id: str | None) -> None:
    _session_id_var.set((session_id or "").strip() or None)


def get_resource_decision() -> dict[str, Any] | None:
    return _decision_var.get()


def get_approved_ops() -> frozenset[str]:
    return _approved_ops_var.get() or frozenset()


def get_resource_session() -> str | None:
    return _session_id_var.get()


def clear_resource_decision() -> None:
    _decision_var.set(None)
    _approved_ops_var.set(None)
    _session_id_var.set(None)


__all__ = [
    "bind_approved_ops",
    "bind_resource_decision",
    "bind_resource_session",
    "clear_resource_decision",
    "get_approved_ops",
    "get_resource_decision",
    "get_resource_session",
]
