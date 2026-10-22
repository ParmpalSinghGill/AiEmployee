"""Platform hardening: shell guardrails, canonical paths, planner enum, titling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from ai_employee.tools.builtins.code import (
    ShellTool,
    _block_unbounded_traversal,
    _strip_stream_redirections,
    _truncate_output,
)

# ── Improvement 3: stream redirection classification ─────────────────────────


def test_strip_stream_redirections() -> None:
    assert _strip_stream_redirections("ls 2>&1") == "ls "
    assert _strip_stream_redirections("cmd 1>&2 2>/dev/null") == "cmd  "
    assert _strip_stream_redirections("grep x f >/dev/null") == "grep x f "
    # Real file-write redirections are preserved.
    assert _strip_stream_redirections("echo hi > out.txt") == "echo hi > out.txt"
    assert _strip_stream_redirections("cmd >> log.txt") == "cmd >> log.txt"


def test_read_only_classifier_ignores_stream_redirections() -> None:
    tool = ShellTool(Path("/tmp"))

    # Descriptor duplication / null sinks are plumbing, not writes.
    assert tool._is_read_only("ls 2>&1") is True
    assert tool._is_read_only("ls 2>/dev/null") is True
    assert tool._is_read_only("grep foo bar >/dev/null 2>&1") is True

    # Actual file writes still require approval.
    assert tool._is_read_only("echo hi > out.txt") is False
    assert tool._is_read_only("cat a b >> log.txt") is False
    # A pipe is still a pipe (not auto-approved as read-only).
    assert tool._is_read_only("ls 2>&1 | head") is False


def test_requires_approval_uses_stripped_classifier() -> None:
    tool = ShellTool(Path("/tmp"))
    assert tool.requires_approval({"command": "ls 2>&1"}) is None
    assert tool.requires_approval({"command": "ls 2>/dev/null"}) is None
    assert tool.requires_approval({"command": "echo data > out.txt"}) == "shell_write"
    assert tool.requires_approval({"command": "cmd >> log.txt"}) == "shell_write"


# ── Improvement 2: shell output truncation ───────────────────────────────────


def test_truncate_lines_keeps_head_and_tail() -> None:
    text = "\n".join(f"line {i}" for i in range(300))
    out = _truncate_output(text)
    # First 150 and last 50 lines are preserved; the middle 100 are omitted.
    assert "line 0" in out
    assert "line 149" in out
    assert "line 299" in out
    assert "line 150" not in out
    assert "output truncated" in out
    assert "100 lines" in out


def test_truncate_chars_caps_long_single_line() -> None:
    text = "x" * 13000
    out = _truncate_output(text)
    assert "output truncated" in out
    assert "1000 chars omitted" in out
    assert len(out.split("\n", 1)[0]) == 12000


def test_truncate_noop_for_small_output() -> None:
    text = "short\noutput\n"
    assert _truncate_output(text) == text
    assert _truncate_output("") == ""


# ── Improvement 2: unbounded traversal guardrail ─────────────────────────────


def test_block_unbounded_traversal() -> None:
    assert _block_unbounded_traversal("find / -name '*.py'") is not None
    assert _block_unbounded_traversal("find /home -name '*.py'") is not None
    assert _block_unbounded_traversal("find /home/user -name '*.py'") is not None
    # A depth bound is allowed.
    assert _block_unbounded_traversal("find /home -maxdepth 2 -name '*.py'") is None
    # A scoped, non-root path is allowed.
    assert _block_unbounded_traversal("find ./data -name '*.py'") is None
    # Non-find commands are untouched.
    assert _block_unbounded_traversal("ls -la /") is None


# ── Improvement 1: canonical absolute paths ──────────────────────────────────


def test_environment_context_lists_absolute_roots(settings) -> None:
    from ai_employee.observability.session_context import build_environment_context

    ctx = build_environment_context(settings)
    assert "Application Root:" in ctx
    assert "Shell Working Directory (cwd):" in ctx
    for line in ctx.splitlines():
        if line.startswith("- Application Root:") or line.startswith("- Shell Working Directory"):
            path = line.split(":", 1)[1].strip()
            assert Path(path).is_absolute(), f"not absolute: {line!r}"


# ── Improvement 4: strict planner enum + raw telemetry ───────────────────────


def test_plan_schema_constrains_agent_to_roster() -> None:
    from ai_employee.planning.planner import _plan_schema_for

    schema = _plan_schema_for(["research", "coding"])
    # Valid names are accepted.
    model = schema.model_validate(
        {
            "goal": "g",
            "steps": [{"id": 1, "description": "d", "agent": "coding"}],
        }
    )
    assert model.steps[0].agent == "coding"
    # A hallucinated name is rejected by the constrained model.
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "goal": "g",
                "steps": [{"id": 1, "description": "d", "agent": "workspace_agent"}],
            }
        )


def test_plan_schema_advertises_enum_in_json_schema() -> None:
    import json

    from ai_employee.planning.planner import _plan_schema_for

    js = _plan_schema_for(["research", "coding"]).model_json_schema()
    dumped = json.dumps(js)
    assert '"enum"' in dumped
    assert '"research"' in dumped
    assert '"coding"' in dumped


async def test_planner_logs_raw_agent_name_when_invalid() -> None:
    from ai_employee.planning.planner import Plan, Planner, PlanStep

    class _FakeBrain:
        async def structured_output(self, messages, schema, **kwargs):
            # Simulate a JSON-mode provider ignoring the enum and emitting a bogus name.
            return Plan(
                goal="g",
                steps=[PlanStep(id=1, description="implement the fix", agent="workspace_agent")],
            )

    catalog = [
        {"name": "research", "role": "Research Analyst", "description": "investigates"},
        {"name": "coding", "role": "Software Engineer", "description": "writes code"},
    ]

    with capture_logs() as logs:
        plan = await Planner(_FakeBrain()).make_plan("fix it", agent_catalog=catalog)

    assert plan.steps[0].agent == "coding"  # corrected from the invalid name
    corrected = [e for e in logs if e["event"] == "planner.corrected_agent"]
    assert corrected, "expected a planner.corrected_agent event"
    # from_agent must be the RAW model output, not the "research" default.
    assert corrected[0]["from_agent"] == "workspace_agent"
    assert corrected[0]["to_agent"] == "coding"


# ── Improvement 5: title generator token budget ──────────────────────────────


async def test_name_task_allocates_enough_output_tokens() -> None:
    from ai_employee.core.employee import AIEmployee

    class _Brain:
        name = "gemini"

        def __init__(self) -> None:
            self.kwargs: dict = {}

        async def complete(self, prompt, **kwargs):
            self.kwargs = kwargs
            return "A Nice Title"

    emp = AIEmployee.__new__(AIEmployee)
    emp.brain = _Brain()  # type: ignore[assignment]
    emp.settings = SimpleNamespace(auto_name_tasks=True)  # type: ignore[attr-defined]

    title = await emp.name_task("some task that needs a title")

    assert title == "A Nice Title"
    assert emp.brain.kwargs.get("max_tokens", 0) >= 256  # type: ignore[union-attr]


# ── Improvement 5: resume title deduplication ────────────────────────────────


def test_session_named_flag_defaults_false() -> None:
    from ai_employee.observability.tasks import _Session

    assert _Session("Raw goal").named is False


async def test_name_session_marks_named_and_sets_title() -> None:
    from ai_employee.observability.tasks import TaskManager, TaskRecord, _Session

    tm = TaskManager()
    session = _Session("Raw goal title")
    tm._sessions[session.id] = session
    task = TaskRecord(goal="some goal", session_id=session.id)
    tm._tasks[task.id] = task

    class _Emp:
        async def name_task(self, goal):
            return "Nice Title"

    await tm._name_session(_Emp(), task)

    assert session.title == "Nice Title"
    assert session.named is True


# ── Improvement 6: git single-turn prompt + skill-recall isolation ────────────


def test_git_prompt_instructs_single_chained_command() -> None:
    from ai_employee.agents.specs import spec_by_name

    prompt = spec_by_name()["git"].system_prompt
    assert "SINGLE chained shell command" in prompt
    assert "git rev-parse HEAD" in prompt
    assert "git status --porcelain" in prompt


def test_disable_skill_recall_setting_defaults_false(settings) -> None:
    assert settings.disable_skill_recall is False


async def test_skill_recall_is_skipped_when_disabled(monkeypatch) -> None:
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.brains.providers.mock import MockBrain
    from ai_employee.config import reload_settings

    class _FakeMemory:
        def __init__(self) -> None:
            self.recall_called = False
            self.find_skill_called = False

        async def find_knowledge(self, *args, **kwargs):
            self.recall_called = True
            return []

        async def find_skill(self, *args, **kwargs):
            self.find_skill_called = True
            return []

    class _Agent(ToolUsingAgent):
        name = "tester"
        role = "tester"
        system_prompt = "sys"
        allowed_tools: list[str] = []
        max_iterations = 1

    settings = reload_settings()
    settings.use_stored_memory = True
    settings.disable_skill_recall = True
    monkeypatch.setattr("ai_employee.agents.base.get_settings", lambda: settings)

    mem = _FakeMemory()
    ctx = AgentContext(brain=MockBrain(), tools=None, memory=mem, session_id="s")  # type: ignore[arg-type]

    await _Agent()._build_initial_messages("task", ctx)

    assert mem.recall_called is True  # long-term recall still runs
    assert mem.find_skill_called is False  # skills are bypassed for zero-shot eval
