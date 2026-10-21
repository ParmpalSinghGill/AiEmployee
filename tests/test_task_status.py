"""A run's terminal status must follow the evidence, not just "no approval".

Previously any pipeline that returned without a pending approval was marked
``completed`` — even when every executed step failed verification. These tests
pin the evidence-based rule in both the live runner and snapshot reload.
"""

from __future__ import annotations

import asyncio

from ai_employee.observability.tasks import TaskManager, TaskRecord, TaskStatus


class _StepsEmployee:
    def __init__(self, steps, *, trace=None) -> None:
        self.steps = steps
        self.trace = trace or []

    async def run(self, goal, **_kwargs):
        return {
            "answer": "synthesized anyway",
            "confidence": 0.6,
            "needs_approval": None,
            "steps": self.steps,
            "trace": self.trace,
        }

    async def name_task(self, goal):
        return "Steps task"

    def persist_memory(self):
        pass


async def _wait_status(tm: TaskManager, task_id: str, status: TaskStatus) -> None:
    for _ in range(200):
        await asyncio.sleep(0.01)
        if tm.get_task(task_id).status is status:
            return
    raise AssertionError(f"task never reached {status}")


async def test_all_failed_steps_mark_task_failed():
    tm = TaskManager(None)
    task = tm.start(
        _StepsEmployee(
            [
                {"id": 1, "agent": "coding", "verified": False},
                {"id": 2, "agent": "qa", "verified": False},
            ]
        ),
        "fix the failing test",
    )

    await _wait_status(tm, task.id, TaskStatus.failed)
    rec = tm.get_task(task.id)
    assert "failed verification" in (rec.error or "")
    assert rec.result is not None


async def test_one_verified_step_still_completes():
    tm = TaskManager(None)
    task = tm.start(
        _StepsEmployee(
            [
                {"id": 1, "agent": "coding", "verified": True},
                {"id": 2, "agent": "qa", "verified": False},
            ]
        ),
        "mostly working",
    )

    await _wait_status(tm, task.id, TaskStatus.completed)
    assert tm.get_task(task.id).error is None


async def test_skipped_steps_do_not_fail_a_run():
    tm = TaskManager(None)
    task = tm.start(
        _StepsEmployee(
            [
                {"id": 1, "agent": "coding", "verified": True},
                {"id": 2, "agent": "qa", "verified": False, "skipped": True},
            ]
        ),
        "one real step",
    )

    await _wait_status(tm, task.id, TaskStatus.completed)


async def test_no_step_evidence_completes():
    """Quick-answer / direct runs have no step records and must stay successful."""
    tm = TaskManager(None)
    task = tm.start(_StepsEmployee([]), "just answer this")
    await _wait_status(tm, task.id, TaskStatus.completed)


def test_snapshot_reload_marks_all_failed_run_failed():
    rec = TaskRecord.from_snapshot(
        {
            "id": "failedrun01",
            "goal": "g",
            "session_id": "s1",
            "status": "running",
            "events": [],
            "result": {
                "answer": "synthesized",
                "needs_approval": None,
                "steps": [
                    {"id": 1, "agent": "git", "verified": False},
                    {"id": 2, "agent": "coding", "verified": False},
                ],
            },
        }
    )

    assert rec.status is TaskStatus.failed
    assert "failed verification" in (rec.error or "")


def test_snapshot_reload_corrects_stored_completed_status():
    """Older snapshots store 'completed' for all-failed runs; correct them on load."""
    rec = TaskRecord.from_snapshot(
        {
            "id": "failedrun02",
            "goal": "g",
            "session_id": "s1",
            "status": "completed",
            "events": [],
            "result": {
                "answer": "synthesized",
                "needs_approval": None,
                "steps": [
                    {"id": 1, "agent": "git", "verified": True},
                    {"id": 2, "agent": "coding", "verified": False},
                ],
            },
        }
    )
    # A mixed run that produced something verified stays completed.
    assert rec.status is TaskStatus.completed

    rec2 = TaskRecord.from_snapshot(
        {
            "id": "failedrun03",
            "goal": "g",
            "session_id": "s1",
            "status": "completed",
            "events": [],
            "result": {
                "answer": "synthesized",
                "needs_approval": None,
                "steps": [
                    {"id": 1, "agent": "git", "verified": False},
                    {"id": 2, "agent": "coding", "verified": False},
                ],
            },
        }
    )
    assert rec2.status is TaskStatus.failed
    assert "failed verification" in (rec2.error or "")
