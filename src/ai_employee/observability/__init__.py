"""Observability — live task tracking for the web UI.

A :class:`TaskManager` runs goals as concurrent background tasks, captures every
agent/tool/verifier event (by observing the structured log stream and attributing
events to the running task via the ``task_id`` contextvar), and exposes snapshots
for the dashboard. Sessions group an initial task with its follow-ups so memory
and context carry across a conversation.
"""

from ai_employee.observability.tasks import (
    TaskEvent,
    TaskManager,
    TaskRecord,
    TaskStatus,
    get_task_manager,
)

__all__ = [
    "TaskManager",
    "TaskRecord",
    "TaskEvent",
    "TaskStatus",
    "get_task_manager",
]
