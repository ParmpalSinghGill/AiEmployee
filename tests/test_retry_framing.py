"""Verifier retries are framed as continuations and tagged by attempt number.

Two guarantees the Explain view depends on:

1. A retry's input is a *continuation* — the original planner step, then the
   previous attempt's result, then the reviewer's issues — not a fresh, rewritten
   task that hides where the retry began.
2. Every trace event emitted during an attempt carries ``retry=N`` (0 = initial),
   so the dashboard can separate attempt 1 from retry 1 instead of blending them.
"""

from __future__ import annotations

import structlog

from ai_employee.agents import AgentRegistry, AgentResult
from ai_employee.memory import MemoryManager
from ai_employee.memory.in_memory import InMemoryVectorStore
from ai_employee.tools.registry import ToolRegistry
from ai_employee.workflows.employee_graph import Orchestrator, _build_retry_task


class _StubAgent:
    def __init__(self, name: str, handler) -> None:
        self.name = name
        self.role = name
        self.description = f"stub {name}"
        self._handler = handler

    async def run(self, task, ctx):
        return await self._handler(task, ctx)


def test_build_retry_task_is_a_continuation() -> None:
    out = _build_retry_task(
        task="ORIGINAL PLANNER STEP",
        previous_output="PREV RESULT",
        issues=["missing test", "wrong path"],
        suggestion="add a test",
        attempt=1,
    )
    assert "ORIGINAL PLANNER STEP" in out
    assert "RETRY 1" in out
    assert "PREV RESULT" in out
    assert "missing test" in out and "wrong path" in out
    assert "add a test" in out
    assert "CONTINUE" in out
    # Order: the real input first, then the previous result, then the issues.
    assert out.index("ORIGINAL PLANNER STEP") < out.index("PREV RESULT") < out.index("missing test")


def test_build_retry_task_handles_empty_previous_output() -> None:
    out = _build_retry_task(
        task="T", previous_output="", issues=[], suggestion="", attempt=2
    )
    assert "RETRY 2" in out
    assert "(the previous attempt produced no output)" in out
    assert "(none listed)" in out


async def test_retry_is_tagged_and_carries_previous_output(brain, settings) -> None:
    seen: list[tuple[str, int | None]] = []

    async def planner_run(goal, ctx):
        return AgentResult(
            agent="planner",
            metadata={
                "plan": {
                    "steps": [
                        {
                            "id": 1,
                            "agent": "worker",
                            "description": "Do the thing",
                            "success_criteria": "done",
                        }
                    ]
                }
            },
        )

    async def worker_run(task, ctx):
        retry = structlog.contextvars.get_contextvars().get("retry")
        seen.append((task, retry))
        return AgentResult(agent="worker", output="first attempt output", confidence=0.5)

    async def ceo_run(goal, ctx):
        return AgentResult(agent="ceo", output="final answer", confidence=0.8)

    agents = AgentRegistry()
    agents.register(_StubAgent("planner", planner_run))
    agents.register(_StubAgent("worker", worker_run))
    agents.register(_StubAgent("ceo", ceo_run))

    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=MemoryManager(InMemoryVectorStore(), brain),
        max_retries_per_step=1,
    )

    class _FlakyVerifier:
        def __init__(self) -> None:
            self.n = 0

        async def verify(self, **kwargs):
            from ai_employee.verification.verifier import Verdict

            self.n += 1
            if self.n == 1:
                return Verdict(
                    passed=False, confidence=0.3, issues=["missing test"], suggestion="add a test"
                )
            return Verdict(passed=True, confidence=0.9)

    orch._verifier = _FlakyVerifier()  # type: ignore[assignment]

    await orch.run("do the thing", session_id="retry-test")

    assert len(seen) == 2, "expected the step to run twice (initial + one retry)"
    # Initial attempt: retry=0, the real planner step, untouched.
    assert seen[0][1] == 0
    assert "Do the thing" in seen[0][0]
    assert "RETRY" not in seen[0][0]
    # Retry attempt: retry=1, a continuation that carries the previous result + issues.
    assert seen[1][1] == 1
    assert "RETRY 1" in seen[1][0]
    assert "CONTINUE" in seen[1][0]
    assert "first attempt output" in seen[1][0]
    assert "missing test" in seen[1][0]


async def test_retry_accumulates_issues_across_attempts(brain, settings) -> None:
    """A later retry carries every prior review's issues, not just the latest one."""
    seen_tasks: list[str] = []

    async def planner_run(goal, ctx):
        return AgentResult(
            agent="planner",
            metadata={
                "plan": {
                    "steps": [
                        {
                            "id": 1,
                            "agent": "worker",
                            "description": "Do the thing",
                            "success_criteria": "done",
                        }
                    ]
                }
            },
        )

    async def worker_run(task, ctx):
        seen_tasks.append(task)
        return AgentResult(agent="worker", output=f"output {len(seen_tasks)}", confidence=0.5)

    async def ceo_run(goal, ctx):
        return AgentResult(agent="ceo", output="final answer", confidence=0.8)

    agents = AgentRegistry()
    agents.register(_StubAgent("planner", planner_run))
    agents.register(_StubAgent("worker", worker_run))
    agents.register(_StubAgent("ceo", ceo_run))

    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=MemoryManager(InMemoryVectorStore(), brain),
        max_retries_per_step=2,
    )

    class _FlakyVerifier:
        def __init__(self) -> None:
            self.n = 0

        async def verify(self, **kwargs):
            from ai_employee.verification.verifier import Verdict

            self.n += 1
            if self.n == 1:
                return Verdict(passed=False, confidence=0.3, issues=["issue one"], suggestion="fix one")
            if self.n == 2:
                return Verdict(passed=False, confidence=0.3, issues=["issue two"], suggestion="fix two")
            return Verdict(passed=True, confidence=0.9)

    orch._verifier = _FlakyVerifier()  # type: ignore[assignment]

    await orch.run("do the thing", session_id="retry-accumulate")

    assert len(seen_tasks) == 3, "expected initial run + two retries"
    # The second retry (attempt 2) must carry BOTH prior issues — not just "issue two".
    assert "issue one" in seen_tasks[2]
    assert "issue two" in seen_tasks[2]
