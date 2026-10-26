"""Approval scope: "this command only" vs "all of this kind for the task".

Approving one paused shell command used to add the whole ``shell_write`` key to
the task, so every later file-changing command ran unasked. A command approval
now pins the exact call; an operation approval still covers the whole task.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ai_employee.tools.approvals import (
    approved_call_token,
    call_tokens_for_approval,
    is_call_token,
)

LS = {"command": "ls -la | head -5"}


# ── tokens ──────────────────────────────────────────────────────────────────


def test_token_pins_tool_and_arguments() -> None:
    token = approved_call_token("shell", LS)
    assert is_call_token(token)
    assert token == approved_call_token("shell", {"command": "ls -la | head -5"})
    assert token == approved_call_token("shell", {**LS, "timeout": 120})  # timeout ignored
    assert token != approved_call_token("shell", {"command": "rm -rf build"})
    assert token != approved_call_token("python_exec", LS)


def test_tokens_cover_the_shown_call_and_its_batch() -> None:
    approval = {
        "operation": "shell_write",
        "tool": "shell",
        "args": LS,
        "batched_calls": [
            {"tool": "shell", "args": LS},
            {"tool": "write_file", "args": {"path": "a.txt", "content": "x"}},
        ],
    }
    tokens = call_tokens_for_approval(approval)
    assert tokens == {
        approved_call_token("shell", LS),
        approved_call_token("write_file", {"path": "a.txt", "content": "x"}),
    }


# ── what an approval click grants ───────────────────────────────────────────


def test_command_scope_never_grants_the_operation_key() -> None:
    from ai_employee.observability.tasks import _command_approval

    granted = _command_approval(
        "shell_write", {"operation": "shell_write", "tool": "shell", "args": LS}
    )
    assert granted == {approved_call_token("shell", LS)}
    assert "shell_write" not in granted


def test_command_scope_falls_back_when_there_is_no_command() -> None:
    from ai_employee.observability.tasks import _command_approval

    # A resource decision has no command to pin: narrow per-operation grant.
    assert "waiting_for_user" in _command_approval(
        "waiting_for_user", {"operation": "waiting_for_user"}
    )


class _PausingEmployee:
    """Pauses on a shell command until it is approved (by key or by token)."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_ops: list[set[str]] = []

    async def run(self, goal, *, session_id=None, approved_ops=None, **_kw):
        self.calls += 1
        ops = set(approved_ops or ())
        self.seen_ops.append(ops)
        checkpoint = {
            "goal": goal,
            "session_id": session_id or "default",
            "plan": {"steps": [{"id": 1, "agent": "coding", "description": "list files"}]},
            "cursor": 0,
            "step_results": [],
        }
        if "shell_write" not in ops and approved_call_token("shell", LS) not in ops:
            return {
                "answer": "",
                "confidence": 0.0,
                "needs_approval": {"operation": "shell_write", "tool": "shell", "args": LS},
                "workflow_checkpoint": checkpoint,
            }
        return {
            "answer": "done",
            "confidence": 0.9,
            "needs_approval": None,
            "workflow_checkpoint": checkpoint,
        }

    async def name_task(self, goal):
        return "Scope task"

    def persist_memory(self):
        pass


async def _wait(tm, task_id, status, tries=300):
    from ai_employee.observability.tasks import TaskStatus

    for _ in range(tries):
        await asyncio.sleep(0.01)
        if tm.get_task(task_id).status == TaskStatus(status):
            return
    raise AssertionError(f"task never reached {status}")


async def test_approve_this_command_only() -> None:
    from ai_employee.observability.tasks import TaskManager

    tm, emp = TaskManager(None), _PausingEmployee()
    task = tm.start(emp, "list the files")
    await _wait(tm, task.id, "awaiting_approval")

    tm.approve(emp, task.id, approve_scope="command")
    await _wait(tm, task.id, "completed")
    rec = tm.get_task(task.id)
    assert approved_call_token("shell", LS) in rec.approved_ops
    assert "shell_write" not in rec.approved_ops  # other shell writes still ask
    granted = [e for e in rec.events if e.event == "approval.granted"]
    assert granted and granted[-1].fields.get("scope") == "command"


async def test_approve_all_for_this_task() -> None:
    from ai_employee.observability.tasks import TaskManager

    tm, emp = TaskManager(None), _PausingEmployee()
    task = tm.start(emp, "list the files")
    await _wait(tm, task.id, "awaiting_approval")

    tm.approve(emp, task.id, approve_scope="operation")
    await _wait(tm, task.id, "completed")
    assert "shell_write" in tm.get_task(task.id).approved_ops


async def test_old_callers_keep_their_meaning() -> None:
    from ai_employee.observability.tasks import TaskManager

    tm, emp = TaskManager(None), _PausingEmployee()
    task = tm.start(emp, "list the files")
    await _wait(tm, task.id, "awaiting_approval")
    tm.approve(emp, task.id, approve_all_similar=False)  # unticked "allow all"
    await _wait(tm, task.id, "completed")
    assert "shell_write" not in tm.get_task(task.id).approved_ops


# ── the agent loop honours the token, and only for that command ─────────────


class _TwoWrites:
    """Calls the approved command, then a different file-changing command."""

    def __init__(self) -> None:
        self.turn = 0

    async def chat(self, messages, **kwargs):
        from ai_employee.brains.schemas import ChatResult, ToolCall, Usage

        self.turn += 1
        if self.turn == 1:
            call = ToolCall(id="1", name="shell", arguments={"command": "echo hi > a.txt"})
        elif self.turn == 2:
            call = ToolCall(id="2", name="shell", arguments={"command": "echo bye > b.txt"})
        else:
            return ChatResult(content="finished both writes", usage=Usage())
        return ChatResult(content="", tool_calls=[call], usage=Usage())


async def test_agent_runs_only_the_approved_command(settings, tmp_path: Path) -> None:
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.tools.builtins.code import ShellTool
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry

    class _Agent(ToolUsingAgent):
        name = "coding"
        allowed_tools = ["shell"]
        max_iterations = 4

    reg = ToolRegistry(settings)
    reg.register(ShellTool(tmp_path))
    ctx = AgentContext(
        brain=_TwoWrites(),  # type: ignore[arg-type]
        tools=reg,
        approved_ops={approved_call_token("shell", {"command": "echo hi > a.txt"})},
    )
    begin_task_tool_history()
    try:
        result = await _Agent().run("write two files", ctx)
    finally:
        clear_task_tool_history()

    assert (tmp_path / "a.txt").exists()  # the approved command ran
    assert not (tmp_path / "b.txt").exists()  # the different one did not
    assert result.needs_approval and result.needs_approval["operation"] == "shell_write"
    assert result.needs_approval["args"] == {"command": "echo bye > b.txt"}


# ── console wiring ──────────────────────────────────────────────────────────


def test_console_offers_both_scopes() -> None:
    js = (Path(__file__).resolve().parents[1] / "src/ai_employee/api/static/app.js").read_text()
    assert "Approve this command only" in js
    assert 'data-scope="command"' in js and 'data-scope="operation"' in js
    assert "for this task" in js
    # Auto Approve still means everything for the task.
    assert 'approve_scope: "operation"' in js


async def test_api_accepts_approve_scope() -> None:
    from ai_employee.api.ui import ApproveTaskRequest

    assert ApproveTaskRequest(approve_scope="command").approve_scope == "command"
    assert ApproveTaskRequest().approve_scope is None
