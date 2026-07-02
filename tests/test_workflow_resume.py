"""Workflow checkpoint resume — approve & continue survives restarts."""

from __future__ import annotations

import pytest

from ai_employee.agents import AgentRegistry, AgentResult
from ai_employee.memory import MemoryManager
from ai_employee.memory.in_memory import InMemoryVectorStore
from ai_employee.tools.registry import ToolRegistry
from ai_employee.workflows.employee_graph import Orchestrator


class _StubAgent:
    def __init__(self, name: str, role: str, handler):
        self.name = name
        self.role = role
        self.description = f"stub {name}"
        self._handler = handler

    async def run(self, task, ctx):
        return await self._handler(task, ctx)


@pytest.fixture
def resume_orchestrator(brain):
    plan_steps = [
        {
            "id": 1,
            "agent": "coding",
            "description": "run gated shell",
            "success_criteria": "shell ran",
        },
    ]
    plan_invocations: list[int] = []
    coding_invocations: list[set[str]] = []

    async def planner_run(goal, ctx):
        plan_invocations.append(1)
        return AgentResult(
            agent="planner",
            success=True,
            metadata={"plan": {"steps": plan_steps}},
        )

    async def coding_run(task, ctx):
        coding_invocations.append(set(ctx.approved_ops))
        if not ctx.approved_ops:
            return AgentResult(
                agent="coding",
                success=False,
                output="Paused: needs approval for shell_write.",
                confidence=0.0,
                needs_approval={"operation": "shell_write", "tool": "shell"},
            )
        return AgentResult(
            agent="coding",
            success=True,
            output="shell completed",
            confidence=0.95,
        )

    async def ceo_run(goal, ctx):
        return AgentResult(agent="ceo", success=True, output="All done.", confidence=0.9)

    agents = AgentRegistry()
    agents.register(_StubAgent("planner", "planner", planner_run))
    agents.register(_StubAgent("coding", "coding", coding_run))
    agents.register(_StubAgent("ceo", "ceo", ceo_run))

    memory = MemoryManager(InMemoryVectorStore(), brain)
    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=memory,
        max_retries_per_step=0,
    )
    orch._test_plan_invocations = plan_invocations
    orch._test_coding_invocations = coding_invocations
    return orch


def test_route_start_skips_plan_when_resuming():
    assert Orchestrator._route_start({"resume_execute": True, "plan": {"steps": []}}) == "execute"
    assert Orchestrator._route_start({"plan": {"steps": []}}) == "plan"
    assert Orchestrator._route_start({"resume_execute": True}) == "plan"


def test_checkpoint_from_state_serializes_cursor_and_results():
    state = {
        "plan": {"steps": [{"id": 1}]},
        "cursor": 2,
        "step_results": [{"id": 1, "output": "partial"}],
    }
    cp = Orchestrator.checkpoint_from_state(state, goal="g", session_id="s1")
    assert cp["goal"] == "g"
    assert cp["session_id"] == "s1"
    assert cp["cursor"] == 2
    assert cp["step_results"][0]["output"] == "partial"


def test_taskrecord_from_snapshot_preserves_awaiting_approval():
    from ai_employee.observability.tasks import TaskRecord, TaskStatus

    rec = TaskRecord.from_snapshot({
        "id": "t1", "goal": "g", "session_id": "s1", "parent_id": None,
        "status": "awaiting_approval",
        "events": [{"seq": 1, "ts": "t", "event": "agent.needs_approval", "fields": {}}],
        "last_seq": 1,
        "result": {
            "needs_approval": {"operation": "shell_write"},
            "workflow_checkpoint": {
                "goal": "g", "session_id": "s1",
                "plan": {"steps": [{"id": 1}]}, "cursor": 0, "step_results": [],
            },
        },
        "error": None,
    })
    assert rec.status == TaskStatus.awaiting_approval
    assert rec.result["workflow_checkpoint"]["cursor"] == 0


async def test_orchestrator_halts_with_checkpoint_then_resumes(resume_orchestrator):
    orch = resume_orchestrator
    first = await orch.run("gated goal", session_id="sess-1")
    assert first.get("needs_approval")
    assert first.get("cursor") == 0
    assert orch._test_plan_invocations == [1]
    assert orch._test_coding_invocations == [set()]

    checkpoint = Orchestrator.checkpoint_from_state(first, goal="gated goal", session_id="sess-1")
    second = await orch.run(
        "gated goal",
        session_id="sess-1",
        approved_ops={"shell_write"},
        workflow_checkpoint=checkpoint,
    )
    assert second.get("needs_approval") is None
    assert second.get("final")
    # Resume must not re-plan; coding runs once with approval granted.
    assert orch._test_plan_invocations == [1]
    assert orch._test_coding_invocations == [set(), {"shell_write"}]
