"""Agent registry, planner and verification tests."""

from __future__ import annotations

from ai_employee.agents import AgentContext, build_agent_registry
from ai_employee.planning.planner import Plan, Planner
from ai_employee.tools import default_registry
from ai_employee.verification import Verifier


def test_registry_has_all_agents():
    reg = build_agent_registry()
    names = set(reg.names())
    for expected in ("planner", "ceo", "research", "coding", "security", "human_approval"):
        assert expected in names
    assert len(names) >= 20


async def test_planner_produces_plan(brain):
    reg = build_agent_registry()
    planner = Planner(brain)
    plan = await planner.make_plan("Build a website", agent_catalog=reg.catalog())
    assert isinstance(plan, Plan)
    assert plan.steps
    valid = set(reg.names())
    assert all(s.agent in valid for s in plan.steps)


async def test_tool_using_agent_runs(brain, settings):
    reg = build_agent_registry()
    agent = reg.get("research")
    ctx = AgentContext(brain=brain, tools=default_registry(settings))
    ctx.scratch["agent_catalog"] = reg.catalog()
    result = await agent.run("Summarize the topic", ctx)
    assert result.success
    assert 0.0 <= result.confidence <= 1.0


async def test_verifier_returns_verdict(brain):
    verifier = Verifier(brain)
    verdict = await verifier.verify(task="Say hi", output="hi", success_criteria="greets")
    assert verdict.confidence >= 0.0


async def test_agent_skips_duplicate_tool_calls(settings):
    """Identical repeat tool calls are short-circuited by the orchestrator's tool
    history (ToolRegistry) so the agent doesn't re-execute the same search."""
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, ToolCall, Usage
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry
    from tests.test_tool_duplicate import EchoTool

    class _DupBrain:
        def __init__(self):
            self.turn = 0

        async def chat(self, messages, **kwargs):
            self.turn += 1
            if self.turn <= 2:
                return ChatResult(
                    content="",
                    tool_calls=[ToolCall(id=str(self.turn), name="echo",
                                         arguments={"query": "cigarette detection"})],
                    usage=Usage(),
                )
            return ChatResult(content="final answer", tool_calls=[], usage=Usage())

    class _DupAgent(ToolUsingAgent):
        name = "dup-tester"
        allowed_tools = ["echo"]
        max_iterations = 5

    reg = ToolRegistry(settings)
    reg.register(EchoTool())
    begin_task_tool_history()
    try:
        result = await _DupAgent().run(
            "find stuff", AgentContext(brain=_DupBrain(), tools=reg)
        )
    finally:
        clear_task_tool_history()

    assert result.success
    assert any(
        "duplicate" in str(t.get("output") or "").lower()
        or "duplicate" in str(t.get("observation") or "").lower()
        for t in result.tool_trace
    )


async def test_duplicate_abort_keeps_toolcalls_paired(settings):
    """When a multi-call turn hits the duplicate-abort limit, every tool_call must
    still get a matching tool response — otherwise the next brain call sends
    dangling tool_call_ids and providers reject it (400)."""
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, Role, ToolCall, Usage
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry
    from tests.test_tool_duplicate import EchoTool

    captured: list[list] = []

    def _call(i):
        return ToolCall(id=f"c{i}", name="echo", arguments={"query": "x"})

    class _Brain:
        def __init__(self):
            self.turn = 0

        async def chat(self, messages, **kwargs):
            captured.append(list(messages))
            self.turn += 1
            if self.turn == 1:
                return ChatResult(content="", tool_calls=[_call(1)], usage=Usage())
            if self.turn in (2, 3):
                base = self.turn * 10
                return ChatResult(content="",
                                  tool_calls=[_call(base), _call(base + 1)], usage=Usage())
            return ChatResult(content="done", tool_calls=[], usage=Usage())

    class _A(ToolUsingAgent):
        name = "dup"
        allowed_tools = ["echo"]
        max_iterations = 8

    reg = ToolRegistry(settings)
    reg.register(EchoTool())
    begin_task_tool_history()
    try:
        result = await _A().run("go", AgentContext(brain=_Brain(), tools=reg))
    finally:
        clear_task_tool_history()

    assert result.success
    for msgs in captured:
        pending: list[str] = []
        for m in msgs:
            if m.role == Role.assistant and m.tool_calls:
                pending += [tc.id for tc in m.tool_calls]
            elif m.role == Role.tool and m.tool_call_id in pending:
                pending.remove(m.tool_call_id)
        assert pending == [], f"dangling tool_call ids (no response): {pending}"
