"""Agent registry, planner and verification tests."""

from __future__ import annotations

from ai_employee.agents import AgentContext, build_agent_registry
from ai_employee.planning.planner import Plan, Planner
from ai_employee.tools import default_registry
from ai_employee.verification import Verifier


def test_planner_prompt_separates_tabular_from_video():
    import inspect

    from ai_employee.planning.planner import Planner

    src = inspect.getsource(Planner.make_plan)
    assert "TABULAR" in src
    assert "IMAGE" in src
    assert "SPEECH" in src
    assert "kind=tabular" in src
    assert "kind=image" in src
    assert "kind=speech" in src


def test_registry_has_all_agents():
    reg = build_agent_registry()
    names = set(reg.names())
    for expected in ("planner", "ceo", "research", "coding", "security", "human_approval"):
        assert expected in names
    assert "resource_downloader" in names
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
                    tool_calls=[
                        ToolCall(
                            id=str(self.turn),
                            name="echo",
                            arguments={"query": "cigarette detection"},
                        )
                    ],
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
        result = await _DupAgent().run("find stuff", AgentContext(brain=_DupBrain(), tools=reg))
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
                return ChatResult(
                    content="", tool_calls=[_call(base), _call(base + 1)], usage=Usage()
                )
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


async def test_resource_acquirer_continues_after_fetch_page_failures(settings):
    """fetch_page 403 must not abort the loop before download_resource is tried."""
    from pydantic import BaseModel

    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, ToolCall, Usage
    from ai_employee.tools.base import Tool, ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry

    class FailPageTool(Tool):
        name = "fetch_page"
        description = "fail"
        category = "research"

        class Args(BaseModel):
            url: str

        async def _run(self, args: Args) -> ToolResult:
            return ToolResult(ok=False, error="HTTP 403 Forbidden")

    class DownloadTool(Tool):
        name = "download_resource"
        description = "ok"
        category = "research"

        class Args(BaseModel):
            url: str
            destination: str = "./x.mp4"

        async def _run(self, args: Args) -> ToolResult:
            return ToolResult(ok=True, output={"path": "./x.mp4"})

    class _Brain:
        def __init__(self):
            self.turn = 0

        async def chat(self, messages, **kwargs):
            self.turn += 1
            if self.turn == 1:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="1", name="fetch_page", arguments={"url": "https://x"}),
                        ToolCall(id="2", name="fetch_page", arguments={"url": "https://y"}),
                    ],
                    usage=Usage(),
                )
            if self.turn == 2:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="3",
                            name="download_resource",
                            arguments={
                                "url": "https://www.youtube.com/watch?v=abc",
                                "destination": "./v.mp4",
                            },
                        )
                    ],
                    usage=Usage(),
                )
            return ChatResult(content="downloaded", tool_calls=[], usage=Usage())

    class _Agent(ToolUsingAgent):
        name = "resource_acquirer"
        allowed_tools = ["fetch_page", "download_resource"]
        max_iterations = 6

    reg = ToolRegistry(settings)
    reg.register(FailPageTool())
    reg.register(DownloadTool())
    begin_task_tool_history()
    try:
        result = await _Agent().run("get mall video", AgentContext(brain=_Brain(), tools=reg))
    finally:
        clear_task_tool_history()

    assert result.success
    assert any(t.get("tool") == "download_resource" and t.get("ok") for t in result.tool_trace)
    assert "aborted after repeated tool failure" not in str(result.metadata.get("note", ""))


async def test_research_agent_continues_after_fetch_page_failures(settings):
    """Research agent should try downloads, not stop after fetch_page 403s."""
    from pydantic import BaseModel

    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, ToolCall, Usage
    from ai_employee.tools.base import Tool, ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry

    class FailPageTool(Tool):
        name = "fetch_page"
        description = "fail"
        category = "research"

        class Args(BaseModel):
            url: str

        async def _run(self, args: Args) -> ToolResult:
            return ToolResult(ok=False, error="HTTP 403 Forbidden")

    class DownloadTool(Tool):
        name = "download_resource"
        description = "ok"
        category = "research"

        class Args(BaseModel):
            url: str
            destination: str = "./session/x.mp4"

        async def _run(self, args: Args) -> ToolResult:
            return ToolResult(ok=True, output={"path": "./sessions/s1/x.mp4"})

    class _Brain:
        def __init__(self):
            self.turn = 0

        async def chat(self, messages, **kwargs):
            self.turn += 1
            if self.turn == 1:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="1", name="fetch_page", arguments={"url": "https://x"}),
                        ToolCall(id="2", name="fetch_page", arguments={"url": "https://y"}),
                    ],
                    usage=Usage(),
                )
            if self.turn == 2:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="3",
                            name="download_resource",
                            arguments={
                                "url": "https://www.youtube.com/watch?v=abc",
                                "destination": "./session/x.mp4",
                            },
                        )
                    ],
                    usage=Usage(),
                )
            return ChatResult(
                content="downloaded exactly one required clip", tool_calls=[], usage=Usage()
            )

    class _Agent(ToolUsingAgent):
        name = "research"
        allowed_tools = ["fetch_page", "download_resource"]
        max_iterations = 6

    reg = ToolRegistry(settings)
    reg.register(FailPageTool())
    reg.register(DownloadTool())
    begin_task_tool_history()
    try:
        result = await _Agent().run(
            "find one mall entrance clip and download it",
            AgentContext(brain=_Brain(), tools=reg),
        )
    finally:
        clear_task_tool_history()

    assert result.success
    assert any(t.get("tool") == "download_resource" and t.get("ok") for t in result.tool_trace)
    assert "aborted after repeated tool failure" not in str(result.metadata.get("note", ""))


async def test_resource_downloader_tries_next_url_after_failed_downloads(settings):
    """Repeated failed download attempts must not stop the agent before later URLs."""
    from pydantic import BaseModel

    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, ToolCall, Usage
    from ai_employee.tools.base import Tool, ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry

    class DownloadTool(Tool):
        name = "download_resource"
        description = "download"
        category = "research"

        class Args(BaseModel):
            url: str
            destination: str = "./x"

        async def _run(self, args: Args) -> ToolResult:
            if args.url.endswith("a"):
                return ToolResult(ok=False, error="HTTP 403 Forbidden")
            if args.url.endswith("b"):
                return ToolResult(ok=False, error="HTTP 403 Forbidden")
            return ToolResult(ok=True, output={"path": "./downloads/c.mp4"})

    class _Brain:
        def __init__(self):
            self.turn = 0

        async def chat(self, messages, **kwargs):
            self.turn += 1
            if self.turn == 1:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            name="download_resource",
                            arguments={"url": "https://example.com/a", "destination": "./a.mp4"},
                        )
                    ],
                    usage=Usage(),
                )
            if self.turn == 2:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="2",
                            name="download_resource",
                            arguments={"url": "https://example.com/b", "destination": "./b.mp4"},
                        )
                    ],
                    usage=Usage(),
                )
            if self.turn == 3:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="3",
                            name="download_resource",
                            arguments={"url": "https://example.com/c", "destination": "./c.mp4"},
                        )
                    ],
                    usage=Usage(),
                )
            return ChatResult(
                content="downloaded the third candidate", tool_calls=[], usage=Usage()
            )

    class _Agent(ToolUsingAgent):
        name = "resource_downloader"
        allowed_tools = ["download_resource"]
        max_iterations = 6

    reg = ToolRegistry(settings)
    reg.register(DownloadTool())
    begin_task_tool_history()
    try:
        result = await _Agent().run(
            "download one of these URLs", AgentContext(brain=_Brain(), tools=reg)
        )
    finally:
        clear_task_tool_history()

    assert result.success
    assert [t.get("ok") for t in result.tool_trace if t.get("tool") == "download_resource"] == [
        False,
        False,
        True,
    ]
    assert "aborted after repeated tool failure" not in str(result.metadata.get("note", ""))


def test_data_analysis_spec_handles_encoding_and_budget():
    from ai_employee.agents.specs import spec_by_name

    spec = spec_by_name()["data_analysis"]
    assert spec.max_iterations >= 12
    prompt = spec.system_prompt.lower()
    assert "encoding" in prompt
    assert "latin-1" in prompt
    assert "tabular" in prompt
    assert "image dataset" in prompt
    assert "speech/audio" in prompt
    assert "list_dir" in (spec.tools or ())
    acquirer = spec_by_name()["resource_acquirer"]
    assert "kind=tabular" in acquirer.system_prompt
    assert "kind=image" in acquirer.system_prompt
    assert "kind=speech" in acquirer.system_prompt


async def test_agent_empty_answer_falls_back_to_tool_trace(settings):
    from pydantic import BaseModel

    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, ToolCall, Usage
    from ai_employee.tools.base import Tool, ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry

    class ProbeTool(Tool):
        name = "python_exec"
        description = "run"
        category = "code"

        class Args(BaseModel):
            code: str = ""

        async def _run(self, args: Args) -> ToolResult:
            return ToolResult(ok=True, output={"stdout": "shape=(9994, 21)\nmissing=11\n"})

    class _Brain:
        def __init__(self):
            self.turn = 0

        async def chat(self, messages, **kwargs):
            self.turn += 1
            if self.turn == 1:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="1", name="python_exec", arguments={"code": "print(1)"})
                    ],
                    usage=Usage(),
                )
            return ChatResult(content="", tool_calls=[], finish_reason="length", usage=Usage())

    class _Agent(ToolUsingAgent):
        name = "data_analysis"
        allowed_tools = ["python_exec"]
        max_iterations = 1

    reg = ToolRegistry(settings)
    reg.register(ProbeTool())
    begin_task_tool_history()
    try:
        result = await _Agent().run("clean the csv", AgentContext(brain=_Brain(), tools=reg))
    finally:
        clear_task_tool_history()

    assert result.output
    assert "shape=(9994, 21)" in result.output
    assert "python_exec" in result.output


async def test_unicodedecode_hint_does_not_abort(settings):
    from pydantic import BaseModel

    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, Role, ToolCall, Usage
    from ai_employee.tools.base import Tool, ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry

    seen_tool_msgs: list[str] = []

    class BoomTool(Tool):
        name = "python_exec"
        description = "run"
        category = "code"

        class Args(BaseModel):
            code: str = ""

        async def _run(self, args: Args) -> ToolResult:
            return ToolResult(
                ok=False,
                error="UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa0 in position 12",
            )

    class _Brain:
        def __init__(self):
            self.turn = 0

        async def chat(self, messages, **kwargs):
            self.turn += 1
            for m in messages:
                if m.role == Role.tool:
                    seen_tool_msgs.append(m.content)
            if self.turn <= 2:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=str(self.turn),
                            name="python_exec",
                            arguments={
                                "code": f"import pandas as pd; pd.read_csv('x{self.turn}.csv')"
                            },
                        )
                    ],
                    usage=Usage(),
                )
            return ChatResult(
                content="Loaded with encoding=latin-1. Cleaned file: data/clean.csv",
                tool_calls=[],
                usage=Usage(),
            )

    class _Agent(ToolUsingAgent):
        name = "data_analysis"
        allowed_tools = ["python_exec"]
        max_iterations = 4

    reg = ToolRegistry(settings)
    reg.register(BoomTool())
    begin_task_tool_history()
    try:
        result = await _Agent().run("load csv", AgentContext(brain=_Brain(), tools=reg))
    finally:
        clear_task_tool_history()

    assert any("latin-1" in msg for msg in seen_tool_msgs)
    assert "aborted after repeated tool failure" not in str(result.metadata.get("note", ""))
    assert "latin-1" in result.output


def test_error_class_normalizes_truncated_tool_json():
    from ai_employee.agents.base import ToolUsingAgent

    err = (
        "invalid arguments: 1 validation error for Args\n"
        "code\n"
        "  Field required [type=missing, input_value={'_raw': '{\"code\": \"impor'}, "
        "input_type=dict]\n"
        "    For further information visit https://errors.pydantic.dev/2.13/v/missing"
    )
    assert ToolUsingAgent._error_class(err) == "InvalidToolArgs"


# ── Truncated-output recovery ────────────────────────────────────────────────
# A response cut off at the token ceiling arrives with finish_reason="length" and
# ends mid-word. The old guard only caught outputs under 20 chars, so a 124-char
# stub ("…auto-downloads it from the official `ult") shipped as the step's answer.


async def test_truncated_answer_triggers_a_complete_retry(settings):
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, Usage
    from ai_employee.tools.registry import ToolRegistry

    calls: list[str] = []

    class _TruncBrain:
        async def chat(self, messages, **kwargs):
            if not calls:
                calls.append("truncated")
                return ChatResult(
                    content="Ultralytics 8.4.87 confirmed; yolov8n.pt is not on disk (the fram",
                    tool_calls=[],
                    finish_reason="length",
                    usage=Usage(),
                )
            calls.append("retry")
            return ChatResult(
                content="Complete: YOLOv8n selected; weights auto-download on first use.",
                tool_calls=[],
                finish_reason="stop",
                usage=Usage(),
            )

    class _A(ToolUsingAgent):
        name = "codesmith"
        allowed_tools: list[str] = []
        max_iterations = 3

    result = await _A().run(
        "write training code", AgentContext(brain=_TruncBrain(), tools=ToolRegistry(settings))
    )

    assert calls == ["truncated", "retry"], "a truncated answer must be re-asked"
    assert result.output.startswith("Complete:")
    assert "(the fram" not in result.output


async def test_complete_answer_is_not_retried(settings):
    """finish_reason='stop' means the model finished — never spend a second call."""
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.schemas import ChatResult, Usage
    from ai_employee.tools.registry import ToolRegistry

    calls: list[str] = []

    class _OkBrain:
        async def chat(self, messages, **kwargs):
            calls.append("call")
            return ChatResult(
                content="A short but complete answer about the chosen model.",
                tool_calls=[],
                finish_reason="stop",
                usage=Usage(),
            )

    class _A(ToolUsingAgent):
        name = "documentation"
        allowed_tools: list[str] = []
        max_iterations = 3

    result = await _A().run(
        "summarize", AgentContext(brain=_OkBrain(), tools=ToolRegistry(settings))
    )
    assert len(calls) == 1
    assert result.output.startswith("A short but complete")
