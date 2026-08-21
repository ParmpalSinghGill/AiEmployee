"""A task stopped mid-run must report 'stopped' and stay resumable.

Before progress checkpointing, ``employee.run`` only produced a checkpoint when the
graph *returned*, so a run cancelled mid-flight saved nothing: the UI showed a
stopped task with no Resume button and every completed step was lost.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from ai_employee.workflows.employee_graph import Orchestrator


def _plan_of(n: int):
    async def fake_plan(self, _state):
        return {
            "plan": {
                "steps": [
                    {
                        "id": i,
                        "description": f"step {i}",
                        "agent": "research",
                        "depends_on": [],
                    }
                    for i in range(1, n + 1)
                ]
            },
            "cursor": 0,
            "step_results": [],
            "trace": [],
        }

    return fake_plan


def _slow_steps(monkeypatch, delay: float):
    """Patch step execution to be slow. Must run BEFORE the employee is built:
    the graph binds its node callables at compile time."""
    orig = Orchestrator._run_one_step

    async def slow(self, state, step, results):
        await asyncio.sleep(delay)
        return await orig(self, state, step, results)

    monkeypatch.setattr(Orchestrator, "_run_one_step", slow)


@pytest.mark.asyncio
async def test_stopped_task_is_marked_stopped_and_resumable(monkeypatch) -> None:
    monkeypatch.setattr(Orchestrator, "_plan_node", _plan_of(3))
    _slow_steps(monkeypatch, 0.5)

    from ai_employee import AIEmployee
    from ai_employee.observability.tasks import TaskManager

    emp = AIEmployee()
    tm = TaskManager()
    try:
        task = tm.start(emp, "long multi-step goal", session_id=None)
        handle = tm._handles.get(task.id)
        await asyncio.sleep(0.8)  # step 1 finished, step 2 in flight
        assert tm.cancel(task.id) is True
        with contextlib.suppress(asyncio.CancelledError):
            await handle

        rec = tm.get_task(task.id)
        snap = rec.snapshot()
        assert rec.status.value == "cancelled"
        assert rec.can_resume(), "a stopped run with saved progress must be resumable"
        assert snap["can_resume"] is True
        # Work done before the stop is preserved, not thrown away.
        assert snap["completed_steps"] >= 1
        assert snap["planned_steps"] == 3
        assert snap["progress_saved_at"] is not None
    finally:
        await emp.aclose()


@pytest.mark.asyncio
async def test_resume_continues_from_checkpoint(monkeypatch) -> None:
    """Resuming must not re-run the steps that already completed."""
    monkeypatch.setattr(Orchestrator, "_plan_node", _plan_of(3))

    ran: list[int] = []
    orig = Orchestrator._run_one_step

    async def tracked(self, state, step, results):
        ran.append(step["id"])
        await asyncio.sleep(0.5)
        return await orig(self, state, step, results)

    monkeypatch.setattr(Orchestrator, "_run_one_step", tracked)

    from ai_employee import AIEmployee
    from ai_employee.observability.tasks import TaskManager

    emp = AIEmployee()
    tm = TaskManager()
    try:
        task = tm.start(emp, "long multi-step goal", session_id=None)
        handle = tm._handles.get(task.id)
        await asyncio.sleep(0.8)
        tm.cancel(task.id)
        with contextlib.suppress(asyncio.CancelledError):
            await handle

        before = list(ran)
        assert before, "at least one step should have run before the stop"

        resumed = tm.resume(emp, task.id)
        assert resumed is not None
        rhandle = tm._handles.get(task.id)
        if rhandle is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await rhandle

        # Step 1 completed before the stop; the resume must not repeat it.
        assert ran.count(1) == 1, f"step 1 re-ran on resume: {ran}"
    finally:
        await emp.aclose()


@pytest.mark.asyncio
async def test_dead_worker_is_reported_as_stopped(monkeypatch) -> None:
    """A 'running' task whose worker vanished must not poll forever."""
    from ai_employee import AIEmployee
    from ai_employee.observability.tasks import TaskManager, TaskStatus

    emp = AIEmployee()
    tm = TaskManager()
    try:
        task = tm.start(emp, "goal", session_id=None)
        handle = tm._handles.get(task.id)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            if handle:
                handle.cancel()
                await handle

        # Simulate a worker that died without reaching the except handlers.
        task.status = TaskStatus.running
        tm._handles.pop(task.id, None)

        rec = tm.get_task(task.id)
        assert rec.status.value == "failed"
        assert "interrupted" in (rec.error or "")
    finally:
        await emp.aclose()
