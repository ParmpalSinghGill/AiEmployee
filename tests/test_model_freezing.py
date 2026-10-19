"""One model per task.

The model the operator had selected when they clicked Run is frozen for the whole
task: every agent in that run uses it. Switching the brain (provider/model) for a
later task must not affect a task that is still running — the retired employee is
closed only once its last in-flight task has finished.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_employee.core.employee import AIEmployee


def _bare_employee() -> AIEmployee:
    """An AIEmployee with its lifecycle flags, but a faked ``aclose``."""
    emp = AIEmployee.__new__(AIEmployee)
    emp._task_refs = 0
    emp._retired = False
    emp._closed = False
    emp._closing = False
    closed: list[bool] = []

    async def fake_aclose() -> None:
        if emp._closed:
            return
        emp._closed = True
        closed.append(True)

    emp.aclose = fake_aclose  # type: ignore[method-assign]
    emp.closed = closed  # type: ignore[attr-defined]
    return emp


# ── Employee lifecycle: defer closing a retired employee ─────────────────────


@pytest.mark.asyncio
async def test_retired_employee_closes_after_last_task_releases() -> None:
    emp = _bare_employee()
    emp.acquire()
    emp.acquire()
    emp.mark_retired()
    await asyncio.sleep(0)
    assert emp.closed == []  # two tasks still reference it

    emp.release()
    await asyncio.sleep(0)
    assert emp.closed == []  # one task still running

    emp.release()  # last task
    await asyncio.sleep(0.01)
    assert emp.closed == [True]


@pytest.mark.asyncio
async def test_retired_with_no_tasks_closes_immediately() -> None:
    emp = _bare_employee()
    emp.mark_retired()
    await asyncio.sleep(0.01)
    assert emp.closed == [True]


@pytest.mark.asyncio
async def test_active_employee_is_not_closed_by_release() -> None:
    emp = _bare_employee()
    emp.acquire()
    emp.release()
    await asyncio.sleep(0.01)
    assert emp.closed == []  # not retired → must stay open


@pytest.mark.asyncio
async def test_close_is_scheduled_only_once() -> None:
    emp = _bare_employee()
    emp.mark_retired()
    emp.release()  # refs already 0; must not schedule a second close
    await asyncio.sleep(0.01)
    assert emp.closed == [True]


# ── The brain is frozen at task launch ───────────────────────────────────────


@pytest.mark.asyncio
async def test_task_manager_freezes_the_brain_at_launch() -> None:
    from ai_employee.observability.tasks import TaskManager

    class _Brain:
        name = "gemini"
        model_name = "gemini-3.8-flash"

    class _Emp:
        brain = _Brain()

        async def run(self, goal, **kw):
            return {
                "final": "ok",
                "needs_approval": None,
                "confidence": 0.8,
                "plan": {},
                "step_results": [],
                "trace": [],
            }

        async def name_task(self, goal):
            return "Title"

        def persist_memory(self):
            pass

        def acquire(self):
            pass

        def release(self):
            pass

    emp = _Emp()
    tm = TaskManager()
    task = tm.start(emp, "goal", session_id=None)

    # The snapshot must record the model that was active when Run was clicked.
    assert task.brain_provider == "gemini"
    assert task.brain_model == "gemini-3.8-flash"
    assert task.snapshot()["brain_provider"] == "gemini"
    assert task.snapshot()["brain_model"] == "gemini-3.8-flash"

    handle = tm._handles.get(task.id)
    if handle is not None:
        await handle
    await asyncio.sleep(0.01)  # let the auto-naming task wind down


@pytest.mark.asyncio
async def test_run_result_reports_the_brain_used() -> None:
    emp = AIEmployee.__new__(AIEmployee)
    emp.brain = SimpleNamespace(name="gemini", model_name="gemini-3.8-flash")  # type: ignore[assignment]

    async def fake_orch_run(goal, **kw):
        return {"final": "ok", "plan": {}, "step_results": [], "trace": [], "confidence": 0.8}

    emp.orchestrator = SimpleNamespace(  # type: ignore[assignment]
        run=fake_orch_run,
        checkpoint_from_state=lambda state, **kw: {},
    )

    result = await emp.run("goal")

    assert result["brain_provider"] == "gemini"
    assert result["brain_model"] == "gemini-3.8-flash"
