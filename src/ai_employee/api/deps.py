"""Shared dependencies — a single process-wide AIEmployee instance."""

from __future__ import annotations

from ai_employee.core.employee import AIEmployee

_employee: AIEmployee | None = None


def get_employee() -> AIEmployee:
    """FastAPI dependency returning the shared employee (lazy-initialized)."""
    global _employee
    if _employee is None:
        _employee = AIEmployee()
    return _employee


async def rebuild_employee() -> AIEmployee:
    """Rebuild the shared employee after a settings change (e.g. switching brain).

    Memory is flushed to disk first and reloaded by the new instance, so accumulated
    context survives the swap. The new brain reflects the updated configuration.

    The retired employee is **not** closed eagerly: a run that started before the
    switch must keep using the model that was selected when it was launched. It is
    marked retired and closed once its last running task finishes (see
    ``AIEmployee.mark_retired`` / ``release``).
    """
    from ai_employee.brains.factory import reset_brain_cache

    global _employee
    old = _employee
    if old is not None:
        old.persist_memory()
    reset_brain_cache()
    _employee = AIEmployee()
    if old is not None:
        old.mark_retired()
    return _employee


async def shutdown_employee() -> None:
    global _employee
    if _employee is not None:
        await _employee.aclose()
        _employee = None
