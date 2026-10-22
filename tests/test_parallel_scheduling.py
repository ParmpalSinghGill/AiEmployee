"""Dependency-aware parallel step scheduling."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from ai_employee.workflows.scheduler import (
    describe_schedule,
    has_runnable_work,
    max_parallelism,
    ready_steps,
    schedule_rounds,
    step_dependencies,
)


def _steps(*specs: tuple[int, list[int]]) -> list[dict]:
    return [
        {"id": sid, "description": f"step {sid}", "agent": "research", "depends_on": deps}
        for sid, deps in specs
    ]


def test_step_dependencies_drops_self_reference() -> None:
    assert step_dependencies({"id": 2, "depends_on": [1, 2]}) == [1]
    assert step_dependencies({"id": 1, "depends_on": None}) == []
    assert step_dependencies({"id": 1, "depends_on": ["x"]}) == []


def test_independent_steps_are_all_ready_at_once() -> None:
    steps = _steps((1, []), (2, []), (3, []))
    assert [s["id"] for s in ready_steps(steps, set())] == [1, 2, 3]
    assert max_parallelism(steps) == 3


def test_dependent_step_waits_then_becomes_ready() -> None:
    steps = _steps((1, []), (2, [1]))
    assert [s["id"] for s in ready_steps(steps, set())] == [1]
    # Step 2 is WAITING until 1 completes, then immediately READY.
    assert [s["id"] for s in ready_steps(steps, {1})] == [2]


def test_running_steps_are_not_rescheduled() -> None:
    steps = _steps((1, []), (2, []))
    assert [s["id"] for s in ready_steps(steps, set(), running_ids={1})] == [2]


def test_diamond_plan_parallelizes_the_middle() -> None:
    #   1 → (2, 3) → 4
    steps = _steps((1, []), (2, [1]), (3, [1]), (4, [2, 3]))
    assert schedule_rounds(steps) == [[1], [2, 3], [4]]
    shape = describe_schedule(steps)
    assert shape["max_parallelism"] == 2
    assert shape["parallelizable"] is True


def test_fully_sequential_plan_is_not_parallelizable() -> None:
    steps = _steps((1, []), (2, [1]), (3, [2]))
    assert schedule_rounds(steps) == [[1], [2], [3]]
    assert describe_schedule(steps)["parallelizable"] is False


def test_unknown_dependency_does_not_strand_a_step() -> None:
    """A planner referencing a step it never emitted must not deadlock the run."""
    steps = _steps((1, []), (2, [99]))
    assert [s["id"] for s in ready_steps(steps, set())] == [1, 2]


def test_dependency_cycle_does_not_deadlock() -> None:
    steps = _steps((1, [2]), (2, [1]))
    waves = schedule_rounds(steps)
    assert sorted(i for w in waves for i in w) == [1, 2]


def test_has_runnable_work_tracks_completion() -> None:
    steps = _steps((1, []), (2, [1]))
    assert has_runnable_work(steps, set()) is True
    assert has_runnable_work(steps, {1}) is True
    assert has_runnable_work(steps, {1, 2}) is False


@pytest.mark.asyncio
async def test_independent_steps_actually_run_concurrently(monkeypatch) -> None:
    """The real payoff: independent steps overlap in time, dependents don't."""
    from ai_employee.workflows.employee_graph import Orchestrator, _StepOutcome

    steps = _steps((1, []), (2, []), (3, [1, 2]))
    state = {"plan": {"steps": steps}, "step_results": [], "session_id": "s"}

    in_flight = 0
    peak = 0
    order: list[int] = []

    async def fake_run_one_step(self, _state, step, _results):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        order.append(step["id"])
        return _StepOutcome(
            record={
                "id": step["id"],
                "description": step["description"],
                "agent": step["agent"],
                "output": f"done {step['id']}",
                "confidence": 0.9,
                "verified": True,
                "retries": 0,
            },
            trace=[{"node": "execute", "step": step["id"]}],
        )

    monkeypatch.setattr(Orchestrator, "_run_one_step", fake_run_one_step)
    orch = Orchestrator.__new__(Orchestrator)

    from ai_employee.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "parallel_steps", True, raising=False)
    monkeypatch.setattr(settings, "max_parallel_steps", 4, raising=False)

    out = await Orchestrator._execute_parallel_node(orch, state)

    assert peak == 2, "steps 1 and 2 should have been in flight together"
    assert order[-1] == 3, "the dependent step must run after its dependencies"
    assert {r["id"] for r in out["step_results"]} == {1, 2, 3}
    assert out["cursor"] == 3


@pytest.mark.asyncio
async def test_parallel_respects_max_concurrency(monkeypatch) -> None:
    from ai_employee.workflows.employee_graph import Orchestrator, _StepOutcome

    steps = _steps((1, []), (2, []), (3, []), (4, []))
    state = {"plan": {"steps": steps}, "step_results": [], "session_id": "s"}

    in_flight = 0
    peak = 0

    async def fake_run_one_step(self, _state, step, _results):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return _StepOutcome(
            record={"id": step["id"], "description": "", "agent": "a", "output": "x"},
            trace=[],
        )

    monkeypatch.setattr(Orchestrator, "_run_one_step", fake_run_one_step)
    from ai_employee.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "parallel_steps", True, raising=False)
    monkeypatch.setattr(settings, "max_parallel_steps", 2, raising=False)

    orch = Orchestrator.__new__(Orchestrator)
    out = await Orchestrator._execute_parallel_node(orch, state)

    assert peak <= 2
    assert len(out["step_results"]) == 4


@pytest.mark.asyncio
async def test_parallel_step_failure_does_not_kill_the_run(monkeypatch) -> None:
    from ai_employee.workflows.employee_graph import Orchestrator, _StepOutcome

    steps = _steps((1, []), (2, []))
    state = {"plan": {"steps": steps}, "step_results": [], "session_id": "s"}

    async def fake_run_one_step(self, _state, step, _results):
        if step["id"] == 1:
            raise RuntimeError("boom")
        return _StepOutcome(
            record={"id": 2, "description": "", "agent": "a", "output": "ok"}, trace=[]
        )

    monkeypatch.setattr(Orchestrator, "_run_one_step", fake_run_one_step)
    from ai_employee.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "parallel_steps", True, raising=False)
    monkeypatch.setattr(settings, "max_parallel_steps", 4, raising=False)

    orch = Orchestrator.__new__(Orchestrator)
    out = await Orchestrator._execute_parallel_node(orch, state)

    by_id = {r["id"]: r for r in out["step_results"]}
    assert by_id[1].get("failed") is True
    assert by_id[2]["output"] == "ok"


@pytest.mark.asyncio
async def test_parallel_halts_on_approval_but_keeps_finished_work(monkeypatch) -> None:
    """An approval gate stops new scheduling; in-flight results are still kept."""
    from ai_employee.workflows.employee_graph import Orchestrator, _StepOutcome

    steps = _steps((1, []), (2, []), (3, [1]))
    state = {"plan": {"steps": steps}, "step_results": [], "session_id": "s"}

    async def fake_run_one_step(self, _state, step, _results):
        if step["id"] == 1:
            return _StepOutcome(
                halt={
                    "needs_approval": {"operation": "download"},
                    "trace": [{"node": "execute", "step": 1, "halted": "approval"}],
                }
            )
        await asyncio.sleep(0.01)
        return _StepOutcome(
            record={"id": step["id"], "description": "", "agent": "a", "output": "ok"},
            trace=[],
        )

    monkeypatch.setattr(Orchestrator, "_run_one_step", fake_run_one_step)
    from ai_employee.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "parallel_steps", True, raising=False)
    monkeypatch.setattr(settings, "max_parallel_steps", 4, raising=False)

    orch = Orchestrator.__new__(Orchestrator)
    out = await Orchestrator._execute_parallel_node(orch, state)

    assert out["needs_approval"] == {"operation": "download"}
    # Step 2 was already running when 1 halted — its work must not be discarded.
    assert [r["id"] for r in out["step_results"]] == [2]
    # Step 3 depended on the halted step, so it never ran.
    assert 3 not in {r["id"] for r in out["step_results"]}


@pytest.mark.asyncio
async def test_parallel_settings_endpoint_roundtrip(tmp_path, monkeypatch) -> None:
    """The ⚡ Parallel badge's API: read state, toggle it, read it back."""
    from httpx import ASGITransport, AsyncClient

    from ai_employee.api.app import create_app
    from ai_employee.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "ui_preferences_file", tmp_path / "prefs.json", raising=False)
    monkeypatch.setattr(settings, "parallel_steps", False, raising=False)
    monkeypatch.setattr(settings, "max_parallel_steps", 4, raising=False)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get("/ui/api/parallel/settings")
        assert r.status_code == 200
        assert r.json()["parallel_steps"] is False

        r = await c.post(
            "/ui/api/parallel/settings",
            json={"parallel_steps": True, "max_parallel_steps": 3, "save_default": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["parallel_steps"] is True
        assert body["max_parallel_steps"] == 3

        r = await c.get("/ui/api/parallel/settings")
        assert r.json()["parallel_steps"] is True
        assert r.json()["saved_default"] is True


@pytest.mark.asyncio
async def test_execute_node_dispatches_on_the_toggle(monkeypatch) -> None:
    """The flag is what selects sequential vs parallel — nothing else."""
    from ai_employee.config import get_settings
    from ai_employee.workflows.employee_graph import Orchestrator

    called = {"parallel": 0}

    async def fake_parallel(self, _state):
        called["parallel"] += 1
        return {"step_results": [], "cursor": 0, "trace": []}

    monkeypatch.setattr(Orchestrator, "_execute_parallel_node", fake_parallel)
    orch = Orchestrator.__new__(Orchestrator)
    settings = get_settings()

    steps = _steps((1, []))
    state = {"plan": {"steps": steps}, "step_results": [], "session_id": "s"}

    monkeypatch.setattr(settings, "parallel_steps", True, raising=False)
    await Orchestrator._execute_node(orch, state)
    assert called["parallel"] == 1

    # Off → the sequential path runs instead (it will call _run_one_step).
    ran = {"seq": 0}

    async def fake_one(self, _s, step, _r):
        from ai_employee.workflows.employee_graph import _StepOutcome

        ran["seq"] += 1
        return _StepOutcome(record={"id": step["id"], "output": "x"}, trace=[])

    monkeypatch.setattr(Orchestrator, "_run_one_step", fake_one)
    monkeypatch.setattr(settings, "parallel_steps", False, raising=False)
    await Orchestrator._execute_node(orch, state)
    assert called["parallel"] == 1
    assert ran["seq"] == 1


@pytest.mark.asyncio
async def test_flow_events_expose_live_agents_midflight(monkeypatch) -> None:
    """The Flow view needs to know which agents are in flight *right now*.

    A step with no tool calls yet is otherwise invisible, so parallel_start /
    parallel_done are the only signal that several agents are working at once.
    """
    from ai_employee.workflows.employee_graph import Orchestrator

    async def fake_plan(self, _state):
        return {
            "plan": {
                "steps": [
                    {"id": 1, "description": "s1", "agent": "scout", "depends_on": []},
                    {"id": 2, "description": "s2", "agent": "research", "depends_on": []},
                    {"id": 3, "description": "s3", "agent": "hardware_analyst", "depends_on": []},
                ]
            },
            "cursor": 0,
            "step_results": [],
            "trace": [],
        }

    monkeypatch.setattr(Orchestrator, "_plan_node", fake_plan)
    orig = Orchestrator._run_one_step

    async def slow(self, state, step, results):
        await asyncio.sleep(0.4)
        return await orig(self, state, step, results)

    monkeypatch.setattr(Orchestrator, "_run_one_step", slow)

    from ai_employee import AIEmployee
    from ai_employee.config import get_settings
    from ai_employee.observability.tasks import TaskManager

    settings = get_settings()
    monkeypatch.setattr(settings, "parallel_steps", True, raising=False)
    monkeypatch.setattr(settings, "max_parallel_steps", 4, raising=False)

    emp = AIEmployee()
    tm = TaskManager()
    try:
        task = tm.start(emp, "parallel goal", session_id=None)
        await asyncio.sleep(0.2)  # all three in flight, none finished

        events = tm.get_task(task.id).snapshot()["events"]
        started = [e for e in events if e["event"] == "workflow.parallel_start"]
        done = [e for e in events if e["event"] == "workflow.parallel_done"]
        sched = [e for e in events if e["event"] == "workflow.parallel_schedule"]

        # Agent names must be on the start event — the flow labels each live node.
        assert {e["fields"]["agent"] for e in started} == {"scout", "research", "hardware_analyst"}
        assert len(started) == 3
        assert not done, "nothing should have finished yet"
        assert sched and sched[0]["fields"]["max_parallelism"] == 3

        handle = tm._handles.get(task.id)
        if handle is not None:
            with contextlib.suppress(Exception):
                await handle
    finally:
        await emp.aclose()


@pytest.mark.asyncio
async def test_parallel_events_are_attributed_to_their_step(monkeypatch) -> None:
    """Every agent event must carry its step id, even when steps run concurrently.

    The Explain view builds one card per step from ``fields.step``. The parallel
    node bound the log context once for the whole node, so nested events had no
    step at all and every one fell through to the synthesize fallback — the tab
    showed only Planner and Synthesize, with no per-agent cards.
    """
    from ai_employee.workflows.employee_graph import Orchestrator

    agents = {1: "coding", 2: "research", 3: "data_analysis"}

    async def fake_plan(self, _state):
        return {
            "plan": {
                "steps": [
                    {"id": i, "description": f"s{i}", "agent": a, "depends_on": []}
                    for i, a in agents.items()
                ]
            },
            "cursor": 0,
            "step_results": [],
            "trace": [],
        }

    monkeypatch.setattr(Orchestrator, "_plan_node", fake_plan)
    orig = Orchestrator._run_one_step

    async def slow(self, state, step, results):
        await asyncio.sleep(0.15)  # guarantee the steps genuinely overlap
        return await orig(self, state, step, results)

    monkeypatch.setattr(Orchestrator, "_run_one_step", slow)

    from ai_employee import AIEmployee
    from ai_employee.config import get_settings
    from ai_employee.observability.tasks import TaskManager

    settings = get_settings()
    monkeypatch.setattr(settings, "parallel_steps", True, raising=False)
    monkeypatch.setattr(settings, "max_parallel_steps", 4, raising=False)

    emp = AIEmployee()
    tm = TaskManager()
    try:
        task = tm.start(emp, "goal", session_id=None)
        handle = tm._handles.get(task.id)
        with contextlib.suppress(Exception):
            await handle

        events = tm.get_task(task.id).snapshot()["events"]
        per_step = [
            e
            for e in events
            if e["event"] in {"agent.start", "agent.context", "agent.answer", "verifier.input"}
        ]
        assert per_step, "no agent events captured"
        # Every one is attributed, so Explain can build a card for it.
        assert all(e["fields"].get("step") is not None for e in per_step)
        assert {e["fields"]["step"] for e in per_step} == {1, 2, 3}

        # And concurrency must not cross-contaminate: each step keeps its own agent.
        for e in events:
            f = e["fields"]
            if e["event"] == "agent.start" and f.get("step") is not None:
                assert agents[f["step"]] == f.get("agent")
    finally:
        await emp.aclose()
