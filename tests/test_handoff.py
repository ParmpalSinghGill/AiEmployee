"""Dependency-aware handoff: a step receives its prerequisites' results, not a
generic "last 3 steps" dump.

The bug: a "fix the error" step (agent 2) re-reproduced the error that a prior
"reproduce the error" step (agent 1) had already documented — because the handoff
was the last three *finished* steps (truncated), not the step's declared
``depends_on``, and the step description still said "reproduce". These tests lock
the fix: the dependent gets exactly its prerequisites' outputs plus an explicit
"do not redo this" instruction.
"""

from __future__ import annotations

from ai_employee.workflows.employee_graph import _HANDOFF_MAX_CHARS, Orchestrator

_REPRO = (
    "REPRODUCE:\n"
    "  python repro.py\n"
    "ERROR:\n"
    "  TypeError: 'OrderBy' object is not subscriptable (compiler.py:2221)\n"
    "FIX: guard expressions in find_ordering_name() before get_order_dir().\n"
    "FILES: /tmp/repro/repro.py, /tmp/repro/traceback.txt"
)


def _rec(id_, output, *, description="", agent="coding", verified=True, **kw):
    return {
        "id": id_,
        "description": description,
        "agent": agent,
        "output": output,
        "verified": verified,
        **kw,
    }


def test_handoff_passes_declared_prerequisite_and_not_unrelated() -> None:
    results = [
        _rec(1, _REPRO, description="Reproduce the crash", agent="coding"),
        _rec(2, "UNRELATED SEARCH RESULTS", description="find a dataset", agent="scout"),
    ]
    step = {"id": 3, "depends_on": [1]}
    handoff = Orchestrator._format_prior(step, results)

    assert "REPRODUCE:" in handoff and "TypeError" in handoff
    assert "do NOT redo or re-reproduce" in handoff
    assert "UNRELATED" not in handoff  # not a dependency → excluded


def test_handoff_empty_when_no_dependencies() -> None:
    results = [_rec(1, _REPRO, description="repro")]
    assert Orchestrator._format_prior({"id": 2, "depends_on": []}, results) == ""


def test_handoff_marks_unverified_and_skips_unusable() -> None:
    results = [
        _rec(1, _REPRO, description="repro", verified=False),
        _rec(2, "(skipped — no usable result)", description="find data", skipped=True),
    ]
    # Unverified but usable → included, flagged unverified.
    handoff = Orchestrator._format_prior({"id": 3, "depends_on": [1]}, results)
    assert "unverified" in handoff and "REPRODUCE:" in handoff
    # Skipped prerequisite → nothing to hand off.
    assert Orchestrator._format_prior({"id": 3, "depends_on": [2]}, results) == ""


def test_handoff_is_bounded() -> None:
    huge = "x" * (_HANDOFF_MAX_CHARS + 100)
    results = [_rec(1, huge, description="big")]
    handoff = Orchestrator._format_prior({"id": 2, "depends_on": [1]}, results)
    # The block is capped, so the prompt can't balloon.
    assert handoff.count("x") <= _HANDOFF_MAX_CHARS


async def test_dependent_step_receives_prerequisite_output(brain, settings) -> None:
    """End-to-end: the 'fix' step's task carries the 'reproduce' step's output."""
    from ai_employee.agents import AgentRegistry, AgentResult
    from ai_employee.memory import MemoryManager
    from ai_employee.memory.in_memory import InMemoryVectorStore
    from ai_employee.tools.registry import ToolRegistry

    seen: dict[str, str] = {}

    class _StubAgent:
        def __init__(self, name, handler):
            self.name = name
            self.role = name
            self.description = f"stub {name}"
            self._handler = handler

        async def run(self, task, ctx):
            return await self._handler(task, ctx)

    async def planner_run(goal, ctx):
        return AgentResult(
            agent="planner",
            metadata={
                "plan": {
                    "steps": [
                        {
                            "id": 1,
                            "agent": "coding",
                            "description": "Reproduce the crash",
                            "depends_on": [],
                            "success_criteria": "traceback captured",
                        },
                        {
                            "id": 2,
                            "agent": "coding",
                            "description": "Fix the crash",
                            "depends_on": [1],
                            "success_criteria": "fixed",
                        },
                    ]
                }
            },
        )

    async def coding_run(task, ctx):
        if "Reproduce the crash" in task:
            seen["repro"] = task
            return AgentResult(agent="coding", output=_REPRO, confidence=0.8)
        seen["fix"] = task
        return AgentResult(agent="coding", output="Fixed the crash", confidence=0.8)

    async def ceo_run(goal, ctx):
        return AgentResult(agent="ceo", output="done", confidence=0.8)

    agents = AgentRegistry()
    agents.register(_StubAgent("planner", planner_run))
    agents.register(_StubAgent("coding", coding_run))
    agents.register(_StubAgent("ceo", ceo_run))

    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=MemoryManager(InMemoryVectorStore(), brain),
        max_retries_per_step=0,
    )
    await orch.run("reproduce and fix", session_id="handoff-test")

    # The fix step got the repro step's output + the "don't redo" instruction.
    assert "Reproduce the crash" in seen["repro"]
    assert "TypeError" in seen["fix"]
    assert "do NOT redo or re-reproduce" in seen["fix"]
    # The repro step itself must NOT have been told to reuse anything.
    assert "do NOT redo" not in seen["repro"]
