"""Opt-in base-agent pinning.

A run may be pinned to a single base agent when the caller explicitly passes
``base_agent=...``. By default (``base_agent=None``) the general ``task``
pipeline is multi-agent: the planner assigns each step to the best specialist.
These tests cover the explicit-pin path and the mode → agent mapping.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_employee.agents.base import AgentContext, AgentResult
from ai_employee.agents.custom import PlannerAgent
from ai_employee.agents.routing import base_agent_for_mode
from ai_employee.core.employee import AIEmployee
from ai_employee.planning.planner import Plan, Planner, PlanStep
from ai_employee.workflows.employee_graph import Orchestrator


def _mixed_plan() -> dict:
    return {
        "goal": "build it",
        "reasoning": "mixed specialists",
        "steps": [
            {"id": 1, "description": "investigate", "agent": "research", "depends_on": []},
            {"id": 2, "description": "write code", "agent": "coding", "depends_on": [1]},
            {"id": 3, "description": "test it", "agent": "testing", "depends_on": [2]},
        ],
    }


class _PlanBrain:
    """Structured-output stub that deliberately ignores the pin instruction."""

    def __init__(self, plan: dict) -> None:
        self.plan = plan
        self.system = ""
        self.messages = []

    async def structured_output(self, messages, schema, **kwargs):
        self.messages = messages
        self.system = messages[0].content
        return Plan.model_validate(self.plan)


# ── mode mapping ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("task", None),
        ("research", "research"),
        ("plan", "planner"),
        ("download", "resource_downloader"),
        (None, None),
        ("unknown-mode", None),
        ("  TASK  ", None),
    ],
)
def test_base_agent_for_mode(mode, expected) -> None:
    assert base_agent_for_mode(mode) == expected


# ── planner ─────────────────────────────────────────────────────────────────
async def test_planner_forces_every_step_onto_the_base_agent() -> None:
    brain = _PlanBrain(_mixed_plan())
    catalog = [
        {"name": "research", "role": "Research Analyst", "description": "investigates"},
        {"name": "coding", "role": "Software Engineer", "description": "writes code"},
        {"name": "testing", "role": "QA", "description": "tests code"},
    ]

    plan = await Planner(brain).make_plan(
        "build it", agent_catalog=catalog, base_agent="research"
    )

    assert [s.agent for s in plan.steps] == ["research", "research", "research"]


async def test_planner_prompt_states_the_pin_and_narrows_the_roster() -> None:
    brain = _PlanBrain(_mixed_plan())
    catalog = [
        {"name": "research", "role": "Research Analyst", "description": "investigates"},
        {"name": "coding", "role": "Software Engineer", "description": "writes code"},
    ]

    await Planner(brain).make_plan("build it", agent_catalog=catalog, base_agent="coding")

    assert "AGENT PINNING" in brain.system
    assert "pinned to `coding`" in brain.system
    assert "- coding:" in brain.system
    # The coding roster line is present; the research roster *line* is not.
    assert "- research:" not in brain.system


async def test_planner_agent_reads_base_agent_from_scratch(monkeypatch) -> None:
    captured: dict = {}

    async def fake_make_plan(self, goal, *, agent_catalog, base_agent=None):
        captured["base_agent"] = base_agent
        return Plan(goal=goal, steps=[PlanStep(id=1, description="x", agent="research")])

    monkeypatch.setattr(Planner, "make_plan", fake_make_plan)
    ctx = AgentContext(brain=None, tools=None)  # type: ignore[arg-type]
    ctx.scratch["agent_catalog"] = [{"name": "research", "role": "r", "description": "d"}]
    ctx.scratch["base_agent"] = "coding"

    await PlannerAgent().run("goal", ctx)

    assert captured["base_agent"] == "coding"


# ── plan node: deterministic override even if the model disobeys ─────────────
class _Registry:
    def __init__(self, planner) -> None:
        self._planner = planner
        self._names = {"research", "coding", "testing"}

    def get(self, name):
        return self._planner

    def __contains__(self, name):
        return name in self._names


async def test_plan_node_overrides_mixed_assignments() -> None:
    class _Planner:
        async def run(self, task, ctx):
            return AgentResult(agent="planner", metadata={"plan": _mixed_plan()})

    orch = Orchestrator.__new__(Orchestrator)
    orch._agents = _Registry(_Planner())  # type: ignore[assignment]
    orch._context = lambda state: SimpleNamespace(scratch={})  # type: ignore[assignment]

    state = await orch._plan_node({"goal": "build it", "base_agent": "research"})

    assert [s["agent"] for s in state["plan"]["steps"]] == ["research", "research", "research"]


async def test_plan_node_without_a_base_agent_keeps_the_plan() -> None:
    class _Planner:
        async def run(self, task, ctx):
            return AgentResult(agent="planner", metadata={"plan": _mixed_plan()})

    orch = Orchestrator.__new__(Orchestrator)
    orch._agents = _Registry(_Planner())  # type: ignore[assignment]
    orch._context = lambda state: SimpleNamespace(scratch={})  # type: ignore[assignment]

    state = await orch._plan_node({"goal": "build it"})

    assert [s["agent"] for s in state["plan"]["steps"]] == ["research", "coding", "testing"]


# ── state / checkpoint plumbing ─────────────────────────────────────────────
async def test_run_puts_base_agent_into_initial_state() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    captured: dict = {}

    async def fake_ainvoke(initial, config):
        captured.update(initial)
        return initial

    orch._graph = SimpleNamespace(ainvoke=fake_ainvoke)  # type: ignore[assignment]
    orch._agents = _Registry(None)  # type: ignore[assignment]

    await orch.run("goal", base_agent="coding")

    assert captured["base_agent"] == "coding"


async def test_resume_keeps_the_plans_step_agents() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    captured: dict = {}

    async def fake_ainvoke(initial, config):
        captured.update(initial)
        return initial

    orch._graph = SimpleNamespace(ainvoke=fake_ainvoke)  # type: ignore[assignment]
    orch._agents = _Registry(None)  # type: ignore[assignment]
    checkpoint = {
        "goal": "goal",
        "session_id": "session",
        "base_agent": "research",  # stale value from a pre-multi-agent run
        "plan": {
            "steps": [
                {"id": 1, "agent": "coding", "description": "x", "depends_on": []},
                {"id": 2, "agent": "testing", "description": "y", "depends_on": [1]},
            ]
        },
        "cursor": 0,
        "step_results": [],
    }

    await orch.run("goal", workflow_checkpoint=checkpoint)

    # The plan is authoritative on resume — a stale base_agent must NOT overwrite
    # the specialists the plan already assigned.
    assert [s["agent"] for s in captured["plan"]["steps"]] == ["coding", "testing"]


def test_checkpoint_carries_base_agent() -> None:
    cp = Orchestrator.checkpoint_from_state(
        {"base_agent": "coding", "plan": {"steps": []}, "cursor": 0},
        goal="g",
        session_id="s",
    )
    assert cp["base_agent"] == "coding"


# ── employee entry point: API/CLI runs are pinned too ────────────────────────
def _fake_employee(captured: dict):
    emp = AIEmployee.__new__(AIEmployee)
    emp.brain = SimpleNamespace(name="mock", model_name="mock")  # type: ignore[assignment]

    async def fake_run(goal, **kwargs):
        captured.update(kwargs)
        return {"final": "", "plan": {}, "step_results": [], "trace": []}

    emp.orchestrator = SimpleNamespace(  # type: ignore[assignment]
        run=fake_run,
        checkpoint_from_state=lambda state, **kw: {},
    )
    return emp


async def test_employee_run_does_not_pin_by_default() -> None:
    captured: dict = {}
    await _fake_employee(captured).run("goal")
    assert captured.get("base_agent") is None


async def test_employee_run_honours_an_explicit_base_agent() -> None:
    captured: dict = {}
    await _fake_employee(captured).run("goal", base_agent="coding")
    assert captured["base_agent"] == "coding"
