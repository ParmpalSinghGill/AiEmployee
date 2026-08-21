"""Duplicate tool-call prevention (orchestrator task scope)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.invocation_history import (
    begin_task_tool_history,
    clear_task_tool_history,
    duplicate_payload,
)
from ai_employee.tools.registry import ToolRegistry


class _EchoArgs(BaseModel):
    query: str


class EchoTool(Tool):
    name = "echo"
    description = "echo args"
    category = "test"
    Args = _EchoArgs

    async def _run(self, args: _EchoArgs) -> ToolResult:
        return ToolResult(ok=True, output={"query": args.query})


class FailTool(Tool):
    name = "fail_once"
    description = "fails"
    category = "test"
    Args = _EchoArgs

    async def _run(self, args: _EchoArgs) -> ToolResult:
        return ToolResult(ok=False, error="nope")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


@pytest.mark.asyncio
async def test_registry_blocks_duplicate_tool_calls(registry: ToolRegistry) -> None:
    begin_task_tool_history()
    try:
        first = await registry.execute("echo", {"query": "people counting dataset"})
        second = await registry.execute("echo", {"query": "people counting dataset"})
    finally:
        clear_task_tool_history()

    assert first.ok
    assert first.metadata.get("duplicate") is not True
    assert second.metadata.get("duplicate") is True
    assert second.output["status"] == "duplicate"
    assert second.output["previous_result"] == {"query": "people counting dataset"}
    # The message now also names a concrete next step instead of only saying "no".
    assert "already ran" in second.output["message"]


@pytest.mark.asyncio
async def test_registry_allows_same_tool_different_args(registry: ToolRegistry) -> None:
    begin_task_tool_history()
    try:
        a = await registry.execute("echo", {"query": "one"})
        b = await registry.execute("echo", {"query": "two"})
    finally:
        clear_task_tool_history()

    assert a.metadata.get("duplicate") is not True
    assert b.metadata.get("duplicate") is not True


@pytest.mark.asyncio
async def test_history_records_successful_calls_only(registry: ToolRegistry) -> None:
    registry.register(FailTool())
    begin_task_tool_history()
    try:
        r1 = await registry.execute("fail_once", {"query": "q"})
        r2 = await registry.execute("fail_once", {"query": "q"})
    finally:
        clear_task_tool_history()

    assert not r1.ok
    assert not r2.ok
    assert r2.metadata.get("duplicate") is not True


@pytest.mark.asyncio
async def test_no_history_outside_task_scope(registry: ToolRegistry) -> None:
    clear_task_tool_history()
    a = await registry.execute("echo", {"query": "outside"})
    b = await registry.execute("echo", {"query": "outside"})
    assert a.metadata.get("duplicate") is not True
    assert b.metadata.get("duplicate") is not True


def test_duplicate_payload_shape() -> None:
    prev = ToolResult(ok=True, output=[{"title": "hit"}])
    payload = duplicate_payload(prev)
    assert payload["status"] == "duplicate"
    assert payload["previous_result"] == [{"title": "hit"}]


# ── Dedup scope ──────────────────────────────────────────────────────────────
# Task-wide dedup meant one agent's read permanently blocked every other agent's.
# In a real run, `data_analysis` was blocked from list_dir because an earlier step
# had already listed the same directory; it hit the repeated-duplicate abort and
# gave up before python_exec ever ran, failing verification.


def test_dedup_is_scoped_to_one_agent_turn() -> None:
    from ai_employee.tools.base import ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        bind_dedup_scope,
        reset_dedup_scope,
    )

    history = begin_task_tool_history()
    args = {"path": "data/neu-det"}

    token_a = bind_dedup_scope("resource_acquirer:1")
    history.record("list_dir", args, ToolResult(ok=True, output="listing"))
    # The same agent repeating itself is still a duplicate.
    assert history.get("list_dir", args) is not None
    reset_dedup_scope(token_a)

    # A different agent needing the same read must NOT be blocked.
    token_b = bind_dedup_scope("data_analysis:2")
    assert history.get("list_dir", args) is None
    reset_dedup_scope(token_b)


def test_same_agent_retry_gets_a_clean_slate() -> None:
    """A verifier-triggered retry re-runs the step; it must not inherit blocks."""
    from ai_employee.tools.base import ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        bind_dedup_scope,
        reset_dedup_scope,
    )

    history = begin_task_tool_history()
    args = {"path": "."}

    t1 = bind_dedup_scope("documentation:attempt-1")
    history.record("list_dir", args, ToolResult(ok=True, output="x"))
    reset_dedup_scope(t1)

    t2 = bind_dedup_scope("documentation:attempt-2")
    assert history.get("list_dir", args) is None
    reset_dedup_scope(t2)
