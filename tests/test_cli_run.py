"""`ai-employee run` must tell the operator when a run is paused for approval.

Regression coverage for a real gap observed live: a run that hit the approval
gate (e.g. write_file needing shell_write approval) returned an empty "answer",
and the CLI printed a blank, unlabeled panel with no indication anything was
waiting on a human — indistinguishable from a silent failure. The fix checks
`needs_approval` and prints an explicit paused message naming the operation.
"""

from __future__ import annotations

from argparse import Namespace

import ai_employee.core.employee as employee_module
from ai_employee.cli import _run_async


class _StubEmployee:
    def __init__(self, result: dict) -> None:
        self._result = result

    async def run(self, goal, *, session_id="default"):
        return self._result

    async def aclose(self) -> None:
        pass


async def test_run_prints_paused_message_when_needs_approval(capsys, monkeypatch) -> None:
    result = {
        "answer": "",
        "confidence": 0.0,
        "needs_approval": {
            "operation": "shell_write",
            "tool": "write_file",
            "batched_calls": [{"tool": "write_file"}, {"tool": "write_file"}],
        },
    }
    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _StubEmployee(result))

    args = Namespace(command="run", goal="do something", session="cli")
    code = await _run_async(args)

    out = capsys.readouterr().out
    assert code == 0
    assert "shell_write" in out
    assert "paused" in out.lower() or "approval" in out.lower()


async def test_run_prints_normal_result_when_not_paused(capsys, monkeypatch) -> None:
    result = {"answer": "All done.", "confidence": 0.9, "needs_approval": None}
    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _StubEmployee(result))

    args = Namespace(command="run", goal="do something", session="cli")
    code = await _run_async(args)

    out = capsys.readouterr().out
    assert code == 0
    assert "All done." in out
    assert "paused" not in out.lower()


class _UsageStubEmployee:
    def __init__(self, result: dict) -> None:
        self._result = result

    async def run(self, goal, *, session_id="default"):
        return self._result

    async def ask_agent(self, agent, task, *, session_id="default"):
        return self._result

    async def aclose(self) -> None:
        pass


async def test_run_prints_agents_and_tools_used(capsys, monkeypatch) -> None:
    result = {
        "answer": "All done.",
        "confidence": 0.9,
        "needs_approval": None,
        "agents_used": [
            {"agent": "scout", "steps": 1, "tools": ["web_search"], "ok": 1, "failed": 0},
            {"agent": "coding", "steps": 1, "tools": ["write_file"], "ok": 1, "failed": 0},
        ],
        "tools_used": [
            {
                "tool": "web_search",
                "calls": 2,
                "ok": 2,
                "failed": 0,
                "agents": ["scout"],
            },
            {
                "tool": "write_file",
                "calls": 1,
                "ok": 1,
                "failed": 0,
                "agents": ["coding"],
            },
        ],
    }
    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _UsageStubEmployee(result))

    args = Namespace(command="run", goal="do something", session="cli")
    code = await _run_async(args)

    out = capsys.readouterr().out
    assert code == 0
    assert "scout" in out
    assert "coding" in out
    assert "web_search" in out
    assert "write_file" in out
    assert "Agents used" in out
    assert "Tools used" in out


async def test_ask_prints_tools_used(capsys, monkeypatch) -> None:
    result = {
        "agent": "documentation",
        "output": "Documented it.",
        "confidence": 0.8,
        "agents_used": [
            {
                "agent": "documentation",
                "steps": 1,
                "tools": ["read_file"],
                "ok": 1,
                "failed": 0,
            }
        ],
        "tools_used": [
            {
                "tool": "read_file",
                "calls": 1,
                "ok": 1,
                "failed": 0,
                "agents": ["documentation"],
            }
        ],
    }
    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _UsageStubEmployee(result))

    args = Namespace(command="ask", agent="documentation", task="Document a function")
    code = await _run_async(args)

    out = capsys.readouterr().out
    assert code == 0
    assert "Documented it." in out
    assert "read_file" in out
    assert "Tools used" in out


async def test_run_json_flag_prints_structured_usage(capsys, monkeypatch) -> None:
    result = {
        "answer": "Done.",
        "confidence": 0.9,
        "needs_approval": None,
        "agents_used": [
            {"agent": "scout", "steps": 1, "tools": ["web_search"], "ok": 1, "failed": 0}
        ],
        "tools_used": [
            {
                "tool": "web_search",
                "calls": 1,
                "ok": 1,
                "failed": 0,
                "agents": ["scout"],
            }
        ],
    }
    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _UsageStubEmployee(result))

    args = Namespace(command="run", goal="do something", session="cli", json=True)
    code = await _run_async(args)

    out = capsys.readouterr().out
    assert code == 0
    assert '"agents_used"' in out
    assert '"tools_used"' in out
    assert '"web_search"' in out
