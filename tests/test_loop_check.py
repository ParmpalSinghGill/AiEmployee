"""Loop-check: extend-vs-stop when an agent exhausts its iteration budget."""

from __future__ import annotations

from ai_employee.agents.loop_check import LoopCheckService, LoopVerdict, heuristic_check

# ── Heuristic unit tests ─────────────────────────────────────────────────────


def _entry(tool: str, args: dict, ok: bool = True, error: str | None = None) -> dict:
    return {"tool": tool, "args": args, "ok": ok, "error": error, "output": "", "observation": ""}


def test_heuristic_flags_repeated_call():
    trace = [_entry("echo", {"query": "x"}) for _ in range(4)]
    v = heuristic_check(trace=trace, messages=[], window=8)
    assert v.is_loop
    assert v.method == "heuristic"
    assert not v.ambiguous


def test_heuristic_flags_cycle():
    # A,B repeated 4x (>3) is a loop.
    trace = [_entry("a", {"i": 1}), _entry("b", {"i": 1})] * 4
    v = heuristic_check(trace=trace, messages=[], window=8)
    assert v.is_loop


def test_heuristic_tolerates_short_cycle():
    # A,B,A,B (twice) — a normal edit→test→edit→test rhythm — must NOT be a loop.
    trace = [_entry("read_file", {"path": "x"}), _entry("shell", {"command": "pytest"})] * 2
    v = heuristic_check(trace=trace, messages=[], window=8)
    assert not v.is_loop


def test_heuristic_flags_repeated_read_of_same_target():
    trace = [
        _entry("read_file", {"path": "x.py"}),
        _entry("list_dir", {"path": "d"}),
        _entry("read_file", {"path": "x.py"}),
        _entry("read_file", {"path": "x.py"}),
        _entry("read_file", {"path": "x.py"}),
    ]
    v = heuristic_check(trace=trace, messages=[], window=8)
    assert v.is_loop
    assert v.signals["max_same_read"] >= 4


def test_heuristic_reports_progress_when_artifact_written():
    trace = [_entry("read_file", {"path": "x"}), _entry("write_file", {"path": "y"})]
    v = heuristic_check(trace=trace, messages=[], window=8)
    assert not v.is_loop
    assert not v.ambiguous


def test_heuristic_is_ambiguous_on_varied_reads():
    trace = [
        _entry("shell", {"command": "git status"}),
        _entry("shell", {"command": "ls tests"}),
        _entry("shell", {"command": "cat x"}),
    ]
    v = heuristic_check(trace=trace, messages=[], window=8)
    assert not v.is_loop
    assert v.ambiguous


def test_heuristic_no_activity_is_not_a_loop():
    v = heuristic_check(trace=[], messages=[], window=8)
    assert not v.is_loop


# ── Service tests ────────────────────────────────────────────────────────────


async def test_service_disabled_returns_disabled(settings):
    s = settings.model_copy(update={"agent_loop_check_enabled": False})
    svc = LoopCheckService(None, s)
    v = await svc.evaluate(
        agent="testing",
        trace=[_entry("read_file", {"path": "x"})],
        messages=[],
        window=8,
        settings=s,
    )
    assert v.method == "disabled"
    assert not v.is_loop


async def test_service_skips_llm_judge_on_mock_provider(settings):
    """Ambiguous verdicts must stay heuristic offline — no model call, deterministic."""

    class _Boom:
        async def structured_output(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("LLM judge must not be called on the mock provider")

    svc = LoopCheckService(_Boom(), settings)
    trace = [_entry("shell", {"command": f"c{i}"}) for i in range(3)]
    v = await svc.evaluate(agent="testing", trace=trace, messages=[], window=8, settings=settings)
    assert v.method == "heuristic"
    assert not v.is_loop  # safe default: allow the bounded extension


# ── LLM judge (hybrid fallback) ──────────────────────────────────────────────


def _ambiguous_trace():
    return [_entry("shell", {"command": f"c{i}"}) for i in range(3)]


def _judge_settings(settings):
    from ai_employee.config.settings import LLMProvider

    # A real (non-mock) provider enables the judge path.
    return settings.model_copy(update={"llm_provider": LLMProvider.deepseek})


async def test_llm_judge_resolves_ambiguous_as_loop(settings):
    class _Judge:
        def __init__(self, is_loop):
            self.is_loop = is_loop
            self.calls = 0

        async def structured_output(self, messages, schema, **kw):
            self.calls += 1
            return schema(is_loop=self.is_loop, reason="judged")

    s = _judge_settings(settings)
    brain = _Judge(True)
    v = await LoopCheckService(brain, s).evaluate(
        agent="testing", trace=_ambiguous_trace(), messages=[], window=8, settings=s
    )
    assert v.method == "llm"
    assert v.is_loop
    assert brain.calls == 1


async def test_llm_judge_resolves_ambiguous_as_progress(settings):
    class _Judge:
        async def structured_output(self, messages, schema, **kw):
            return schema(is_loop=False, reason="still progressing")

    s = _judge_settings(settings)
    v = await LoopCheckService(_Judge(), s).evaluate(
        agent="testing", trace=_ambiguous_trace(), messages=[], window=8, settings=s
    )
    assert v.method == "llm"
    assert not v.is_loop


async def test_llm_judge_failure_falls_back_to_heuristic(settings):
    class _Boom:
        async def structured_output(self, *a, **k):
            raise RuntimeError("provider down")

    s = _judge_settings(settings)
    v = await LoopCheckService(_Boom(), s).evaluate(
        agent="testing", trace=_ambiguous_trace(), messages=[], window=8, settings=s
    )
    assert v.method == "heuristic"
    assert v.ambiguous  # safe default: allow the bounded extension


async def test_llm_judge_not_called_on_definite_loop(settings):
    class _Judge:
        def __init__(self):
            self.calls = 0

        async def structured_output(self, *a, **k):
            self.calls += 1
            raise AssertionError("must not be called for a definite loop")

    s = _judge_settings(settings)
    brain = _Judge()
    trace = [_entry("read_file", {"path": "x"})] * 4
    v = await LoopCheckService(brain, s).evaluate(
        agent="testing", trace=trace, messages=[], window=8, settings=s
    )
    assert v.is_loop and v.method == "heuristic"
    assert brain.calls == 0


# ── Integration with the agent loop ──────────────────────────────────────────


class _NeverAnswersBrain:
    """Always requests a *distinct* tool call, so it runs to the cap every time."""

    def __init__(self, tool: str = "echo") -> None:
        self.tool = tool
        self.turn = 0

    async def chat(self, messages, **kwargs):
        from ai_employee.brains.schemas import ChatResult, ToolCall, Usage

        self.turn += 1
        return ChatResult(
            content="",
            tool_calls=[
                ToolCall(id=f"c{self.turn}", name=self.tool, arguments={"query": f"q{self.turn}"})
            ],
            usage=Usage(),
        )


def _build_agent(max_iterations: int):
    from ai_employee.agents.base import ToolUsingAgent

    class _Agent(ToolUsingAgent):
        name = "loopy"
        allowed_tools = ["echo"]

    _Agent.max_iterations = max_iterations
    return _Agent()


def _registry(settings):
    from ai_employee.tools.registry import ToolRegistry
    from tests.test_tool_duplicate import EchoTool

    reg = ToolRegistry(settings)
    reg.register(EchoTool())
    return reg


async def test_agent_extends_when_not_looping(settings, monkeypatch):
    """Distinct calls → ambiguous → the agent gets its extra windows, then stops."""
    import ai_employee.agents.base as base_mod
    from ai_employee.agents.base import AgentContext
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )

    s = settings.model_copy(update={"agent_loop_check_enabled": True, "agent_loop_extensions_max": 2})
    monkeypatch.setattr(base_mod, "get_settings", lambda: s)

    agent = _build_agent(max_iterations=3)
    begin_task_tool_history()
    try:
        result = await agent.run(
            "do work", AgentContext(brain=_NeverAnswersBrain(), tools=_registry(s))
        )
    finally:
        clear_task_tool_history()

    # 3 rounds x (1 base + 2 extensions) = 9
    assert result.steps_taken == 9
    checks = result.metadata.get("loop_checks") or []
    assert len(checks) == 2
    assert all(c["is_loop"] is False for c in checks)
    assert "max_iterations" in result.metadata["note"]


async def test_agent_does_not_extend_when_disabled(settings, monkeypatch):
    import ai_employee.agents.base as base_mod
    from ai_employee.agents.base import AgentContext
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )

    s = settings.model_copy(update={"agent_loop_check_enabled": False})
    monkeypatch.setattr(base_mod, "get_settings", lambda: s)

    agent = _build_agent(max_iterations=3)
    begin_task_tool_history()
    try:
        result = await agent.run(
            "do work", AgentContext(brain=_NeverAnswersBrain(), tools=_registry(s))
        )
    finally:
        clear_task_tool_history()

    assert result.steps_taken == 3
    assert not (result.metadata.get("loop_checks") or [])


async def test_agent_stops_when_loop_check_says_stuck(settings, monkeypatch):
    """When the loop check reports a loop, the step stops at the cap (no extension)."""
    import ai_employee.agents.base as base_mod
    from ai_employee.agents.base import AgentContext
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )

    class _StubService:
        def __init__(self, *a, **k) -> None:
            pass

        async def evaluate(self, **kwargs) -> LoopVerdict:
            return LoopVerdict(True, 0.9, "heuristic", "repeating the same call", {})

    s = settings.model_copy(update={"agent_loop_check_enabled": True, "agent_loop_extensions_max": 2})
    monkeypatch.setattr(base_mod, "get_settings", lambda: s)
    monkeypatch.setattr(base_mod, "LoopCheckService", _StubService)

    agent = _build_agent(max_iterations=4)
    begin_task_tool_history()
    try:
        result = await agent.run(
            "do work", AgentContext(brain=_NeverAnswersBrain(), tools=_registry(s))
        )
    finally:
        clear_task_tool_history()

    assert result.steps_taken == 4
    assert "stopped by loop check" in result.metadata["note"]
    assert result.metadata["loop_checks"][0]["is_loop"] is True


# ── A4: settings wiring ──────────────────────────────────────────────────────


def test_settings_expose_loop_check_knobs(monkeypatch):
    """The AGENT_LOOP_* env vars must actually reach Settings (typo guard)."""
    from ai_employee.config import reload_settings

    try:
        monkeypatch.setenv("AGENT_LOOP_EXTENSIONS_MAX", "5")
        monkeypatch.setenv("AGENT_LOOP_CHECK_ENABLED", "false")
        monkeypatch.setenv("AGENT_LOOP_CYCLE_MIN_REPEATS", "6")
        s = reload_settings()
        assert s.agent_loop_extensions_max == 5
        assert s.agent_loop_check_enabled is False
        assert s.agent_loop_cycle_min_repeats == 6
    finally:
        reload_settings()  # drop the overrides from the cache


# ── A5: observability ────────────────────────────────────────────────────────


def test_loop_check_events_are_flow_events():
    """The loop-check milestones must be captured for the live Flow/log view."""
    from ai_employee.observability.tasks import TaskEvent

    for ev in ("agent.loop_check", "agent.loop_extended", "agent.loop_stopped"):
        assert TaskEvent(seq=1, ts="t", event=ev, fields={"agent": "testing"}).is_flow


async def test_loop_checks_persist_into_step_records(brain, settings):
    """The extend/stop decision must survive into the run snapshot (Result view)."""
    from ai_employee.agents import AgentRegistry, AgentResult
    from ai_employee.memory import MemoryManager
    from ai_employee.memory.in_memory import InMemoryVectorStore
    from ai_employee.tools.registry import ToolRegistry
    from ai_employee.workflows.employee_graph import Orchestrator
    from tests.test_workflow_dependencies import _StubAgent

    async def planner_run(goal, ctx):
        return AgentResult(
            agent="planner",
            success=True,
            metadata={
                "plan": {
                    "steps": [
                        {
                            "id": 1,
                            "agent": "testing",
                            "description": "write a regression test",
                            "depends_on": [],
                            "success_criteria": "test exists",
                        }
                    ]
                }
            },
        )

    async def testing_run(task, ctx):
        return AgentResult(
            agent="testing",
            output="Worked for three windows but produced no final answer.",
            confidence=0.4,
            steps_taken=9,
            metadata={
                "note": "max_iterations reached",
                "loop_checks": [
                    {"is_loop": False, "method": "heuristic", "reason": "progress", "confidence": 0.75}
                ],
            },
        )

    async def ceo_run(goal, ctx):
        return AgentResult(agent="ceo", output="done", confidence=0.8)

    agents = AgentRegistry()
    agents.register(_StubAgent("planner", planner_run))
    agents.register(_StubAgent("testing", testing_run))
    agents.register(_StubAgent("ceo", ceo_run))

    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=MemoryManager(InMemoryVectorStore(), brain),
        max_retries_per_step=0,
    )
    state = await orch.run("write a test", session_id="loop-persist")
    record = (state.get("step_results") or [])[0]
    assert record["steps_taken"] == 9
    assert record["loop_checks"][0]["is_loop"] is False
