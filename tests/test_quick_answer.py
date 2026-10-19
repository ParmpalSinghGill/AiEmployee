"""Quick-answer routing + environment context (efficiency improvements).

A short factual question is answered by the research agent in one turn, skipping
the plan→multi-agent→verify→synthesize pipeline. A tiny environment block (workspace,
Python, sibling dirs) is injected up front so agents stop re-discovering it via shell.
"""

from __future__ import annotations

from ai_employee.workflows.employee_graph import Orchestrator, is_quick_question


# ── classifier ───────────────────────────────────────────────────────────────
def test_is_quick_question_true_cases() -> None:
    for goal in (
        "what is the capital of France?",
        "do you update the actual code at AIEmployee_Workspace?",
        "how does the memory system work?",
        "where is the workspace directory?",
        "why is the sky blue?",
        "what does this project do?",
    ):
        assert is_quick_question(goal), goal


def test_is_quick_question_rejects_actions_and_requests() -> None:
    for goal in (
        "write a training script",  # no "?"
        "can you update the README?",  # request
        "please fix the bug?",  # request
        "analyze sales.csv",  # action
        "find a model and a dataset",  # action
        "download the video?",  # action phrased with ?
        "do the thing",  # no "?"
        "say hello",  # no "?"
    ):
        assert not is_quick_question(goal), goal


# ── routing ──────────────────────────────────────────────────────────────────
async def test_quick_question_answered_directly(brain) -> None:
    from ai_employee.agents import AgentRegistry, AgentResult
    from ai_employee.memory import MemoryManager
    from ai_employee.memory.in_memory import InMemoryVectorStore
    from ai_employee.tools.registry import ToolRegistry

    calls: dict[str, str] = {}

    class _Stub:
        def __init__(self, name, handler):
            self.name = name
            self.role = name
            self.description = f"stub {name}"
            self._h = handler

        async def run(self, task, ctx):
            return await self._h(task, ctx)

    async def planner(task, ctx):
        calls["planner"] = task
        raise AssertionError("planner must not run for a quick question")

    async def research(task, ctx):
        calls["research"] = task
        return AgentResult(agent="research", output="42", confidence=0.9)

    agents = AgentRegistry()
    agents.register(_Stub("planner", planner))
    agents.register(_Stub("research", research))

    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=MemoryManager(InMemoryVectorStore(), brain),
    )
    state = await orch.run("what is the answer to life?", session_id="q1")

    assert state["final"] == "42"
    assert state.get("plan") is None
    assert state.get("step_results") == []
    assert "planner" not in calls
    assert "answer to life" in calls["research"]


async def test_normal_goal_still_plans(brain) -> None:
    from ai_employee.agents import AgentRegistry, AgentResult
    from ai_employee.memory import MemoryManager
    from ai_employee.memory.in_memory import InMemoryVectorStore
    from ai_employee.tools.registry import ToolRegistry

    called: list[str] = []

    class _Stub:
        def __init__(self, name, handler):
            self.name = name
            self.role = name
            self.description = f"stub {name}"
            self._h = handler

        async def run(self, task, ctx):
            return await self._h(task, ctx)

    async def planner(task, ctx):
        called.append("planner")
        return AgentResult(
            agent="planner",
            metadata={"plan": {"steps": [{"id": 1, "agent": "research", "description": "do it", "depends_on": []}]}},
        )

    async def research(task, ctx):
        called.append("research")
        return AgentResult(agent="research", output="done", confidence=0.8)

    async def ceo(task, ctx):
        return AgentResult(agent="ceo", output="final", confidence=0.8)

    agents = AgentRegistry()
    agents.register(_Stub("planner", planner))
    agents.register(_Stub("research", research))
    agents.register(_Stub("ceo", ceo))

    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=MemoryManager(InMemoryVectorStore(), brain),
    )
    state = await orch.run("write a report", session_id="q2")

    assert "planner" in called and "research" in called
    assert state.get("plan")  # normal pipeline produced a plan


# ── environment context ──────────────────────────────────────────────────────
def test_environment_context_lists_workspace_python_and_siblings(tmp_path, monkeypatch) -> None:
    from ai_employee.config import get_settings
    from ai_employee.observability.session_context import build_environment_context

    s = get_settings()
    monkeypatch.setattr(s, "workspace_dir", tmp_path / "workspace")
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / "django-11555").mkdir()
    (tmp_path / "ai_employee").mkdir()

    ctx = build_environment_context(s)
    assert "workspace" in ctx
    assert "Python:" in ctx
    assert "django-11555" in ctx  # sibling checkout surfaced, so agents don't `cd .. && ls`
    assert "ENVIRONMENT" in ctx
